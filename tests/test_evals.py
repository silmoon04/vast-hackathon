import asyncio
from types import SimpleNamespace

import pytest

from vastkit import evals


def test_retrieval_metrics():
    assert evals.hit_at_k(["a", "b", "c"], {"c"}, k=3) and not evals.hit_at_k(["a", "b", "c"], {"c"}, k=2)
    assert evals.reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert evals.time_match((10, 12), (13, 15), tol=2) and not evals.time_match((10, 12), (15, 16), tol=2)
    assert evals.temporal_iou((0, 10), (5, 15)) == pytest.approx(1 / 3)
    assert evals.precision_recall_f1(8, 2, 2)["f1"] == pytest.approx(0.8)


def test_intervals():
    lo, hi = evals.wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    p, lo, hi = evals.bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], seed=1)
    assert p == 0.7 and lo < p < hi
    p, lo, hi = evals.bootstrap_ci([1, 1, 0, 0], groups=["a", "a", "b", "b"], seed=1)
    assert p == 0.5 and lo == 0.0 and hi == 1.0
    d, lo, hi = evals.paired_bootstrap_delta([0, 0, 1, 0], [1, 1, 1, 0])
    assert d == 0.5


def test_kappa():
    assert evals.cohen_kappa("aabb", "aabb") == 1.0
    assert evals.cohen_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"]) == 0.0


class _Call:
    """Stand-in for weave's Call: op errors land on .exception, apply_scorer is async."""

    def __init__(self, output, exception=None):
        self.output, self.exception, self.scored = output, exception, 0

    async def apply_scorer(self, scorer):
        self.scored += 1
        return SimpleNamespace(result=scorer(self.output))


class _SyncOp:
    def __init__(self, output, exception=None):
        self.last = _Call(output, exception)

    def call(self, *args, **kwargs):  # weave: (result, call), never raises
        return self.last.output, self.last


class _AsyncOp(_SyncOp):
    async def call(self, *args, **kwargs):  # weave returns a coroutine for async ops
        return self.last.output, self.last


def _guard(op):
    return asyncio.run(evals.guarded_call(op, lambda out: {"ok": out == "good"}, "q",
                                          passed=lambda v: v["ok"], fallback="FALLBACK"))


@pytest.mark.parametrize("op_cls", [_SyncOp, _AsyncOp])
def test_guarded_call_sync_and_async(op_cls):
    assert _guard(op_cls("good")) == ("good", {"ok": True})
    assert _guard(op_cls("bad")) == ("FALLBACK", {"ok": False})


@pytest.mark.parametrize("op_cls", [_SyncOp, _AsyncOp])
def test_guarded_call_failed_op_skips_scorer(op_cls):
    op = op_cls(None, exception='{"type": "HTTPError", "message": "500"}')
    out, verdict = _guard(op)
    assert out == "FALLBACK" and verdict == {"error": op.last.exception} and op.last.scored == 0
