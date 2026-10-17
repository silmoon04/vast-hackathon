"""Generic evaluation plumbing: textbook retrieval metrics, confidence intervals, agreement,
and thin wrappers over Weave's documented API (Dataset / Evaluation / feedback / guardrail).

Weave notes (weave 0.53.12 + openai 3.31, checked on this PC):
- ``weave.init`` autopatches ``client.chat.completions.create`` -> every LLM call is traced.
- The ``beta.chat.completions.parse`` patch fails on openai 3.x: use ``create`` with
  response_format json_schema instead (vastkit.llm does).
- Set WEAVE_PARALLELISM=4..8 for W&B Inference rate limits.
"""
from __future__ import annotations

import inspect
import math
import random
from collections import defaultdict
from typing import Any, Callable, Hashable, Iterable, Sequence


# ---------------------------------------------------------------- retrieval metrics
def hit_at_k(ranked: Sequence[Any], relevant: Iterable[Any] | Callable[[Any], bool], k: int = 10) -> bool:
    ok = relevant if callable(relevant) else (lambda x, rel=set(relevant): x in rel)
    return any(ok(x) for x in ranked[:k])


def reciprocal_rank(ranked: Sequence[Any], relevant: Iterable[Any] | Callable[[Any], bool], k: int = 10) -> float:
    ok = relevant if callable(relevant) else (lambda x, rel=set(relevant): x in rel)
    for i, x in enumerate(ranked[:k], start=1):
        if ok(x):
            return 1.0 / i
    return 0.0


def time_match(pred: tuple[float, float], gold: tuple[float, float], tol: float = 2.0) -> bool:
    """Temporal hit: intervals overlap after widening the prediction by ``tol`` seconds."""
    return pred[0] - tol <= gold[1] and pred[1] + tol >= gold[0]


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


# ---------------------------------------------------------------- intervals & agreement
def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (good for small n)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: Sequence[float], *, stat: Callable[[Sequence[float]], float] | None = None,
                 groups: Sequence[Hashable] | None = None, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """(point, lo, hi). Pass ``groups`` (e.g. segment/video ids) for a cluster bootstrap."""
    stat = stat or (lambda v: sum(v) / len(v) if v else 0.0)
    point = stat(values)
    rng = random.Random(seed)
    if groups is not None:
        by: dict[Hashable, list[float]] = defaultdict(list)
        for v, g in zip(values, groups):
            by[g].append(v)
        keys = list(by)
        draw = lambda: [x for g in (rng.choice(keys) for _ in keys) for x in by[g]]  # noqa: E731
    else:
        draw = lambda: [rng.choice(values) for _ in values]  # noqa: E731
    if not values:
        return (point, 0.0, 0.0)
    stats = sorted(stat(draw()) for _ in range(n_boot))
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (point, lo, hi)


def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float], *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean(b - a) with a percentile CI for paired per-item scores (e.g. baseline vs new)."""
    d = [y - x for x, y in zip(a, b)]
    return bootstrap_ci(d, n_boot=n_boot, seed=seed)


def cohen_kappa(r1: Sequence[Hashable], r2: Sequence[Hashable]) -> float:
    n = len(r1)
    if n == 0:
        return 0.0
    labels = set(r1) | set(r2)
    po = sum(1 for a, b in zip(r1, r2) if a == b) / n
    pe = sum((list(r1).count(lab) / n) * (list(r2).count(lab) / n) for lab in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ---------------------------------------------------------------- weave helpers (lazy import)
def weave_init(project: str | None = None, **settings: Any):
    """weave.init on '<entity>/<project>' from config; returns the client."""
    import weave
    from .config import settings as _settings
    from .http import ensure_no_proxy
    ensure_no_proxy()
    proj = project or _settings().wandb_project_path
    if not proj:
        raise RuntimeError("set WANDB_ENTITY/WANDB_TEAM and WANDB_PROJECT (or pass project=)")
    return weave.init(proj, settings=settings or None)


def publish_dataset(name: str, rows: list[dict[str, Any]]):
    """Publish (or version) a weave.Dataset; returns its ref so evals pin an exact version."""
    import weave
    ds = weave.Dataset(name=name, rows=rows)
    return weave.publish(ds)


def make_evaluation(name: str, dataset: Any, scorers: list[Any], *, trials: int = 1):
    import weave
    return weave.Evaluation(name=name, dataset=dataset, scorers=scorers, trials=trials)


async def guarded_call(op: Any, scorer: Any, *args: Any, passed: Callable[[dict[str, Any]], bool],
                       fallback: Any = None, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Guardrail pattern: run op via .call(), score it inline (stored as feedback), gate the output.

    Works for sync and async ops. ``op.call`` never raises (weave stores the error on the Call), so a
    failed op skips the scorer and returns ``(fallback, {"error": call.exception})``.
    """
    out = op.call(*args, **kwargs)
    result, call = (await out) if inspect.isawaitable(out) else out
    if getattr(call, "exception", None):
        return fallback, {"error": call.exception}
    s = await call.apply_scorer(scorer)
    verdict = s.result if isinstance(s.result, dict) else {"result": s.result}
    return (result if passed(verdict) else fallback), verdict


def add_feedback(call: Any, *, reaction: str | None = None, note: str | None = None,
                 payload: dict[str, Any] | None = None) -> None:
    """Human feedback on a traced call (thumbs, note or structured payload)."""
    if reaction:
        call.feedback.add_reaction(reaction)
    if note:
        call.feedback.add_note(note)
    if payload:
        call.feedback.add("label", payload)
