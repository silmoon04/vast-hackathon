"""Ledger tests on real data (ptz_parking slices + car_detection). Runs in well under 60 s."""
from __future__ import annotations

import json

import numpy as np
import pytest

from stm import data, ledger

REASONS = {"dark", "blur", "frozen", "blocked", "no_signal", "moved", "turned_away"}
KEYS = {"camera", "seconds", "blind", "blind_total_s", "blind_by_reason", "blind_by_day", "home_views",
        "reason_words", "spots", "per_second"}


@pytest.fixture(scope="module")
def ptz60():
    return ledger.sample("ptz_parking", max_s=50)


def _copy(smp):
    return ledger.Samples(smp.cam, smp.seconds, smp.w, smp.h, [None if g is None else g.copy() for g in smp.grey])


def test_real_slice_is_visible(ptz60):
    hl = ledger.health("ptz_parking", ptz60)
    vis = [p["visible"] for p in hl["per_second"]]
    assert len(vis) == 51
    assert all(vis), hl["blind"]                # the first 50 s of the car park are plainly visible


def test_simulated_blackout_is_dark(ptz60):
    """SIMULATED: black out 20 s of real frames in memory; they must be reported blind/dark."""
    smp = _copy(ptz60)
    for s in range(20, 40):
        smp.grey[s] = np.zeros_like(smp.grey[s])
    hl = ledger.health("ptz_parking", smp)
    ps = hl["per_second"]
    assert all(not ps[s]["visible"] and ps[s]["reason"] == "dark" for s in range(20, 40))
    assert all(ps[s]["visible"] for s in range(5, 15))
    dark = [b for b in hl["blind"] if b["reason"] == "dark"]
    assert len(dark) == 1 and dark[0]["start"] == 20.0 and dark[0]["end"] == 40.0
    assert 0.9 <= dark[0]["score"] <= 1.0


def test_simulated_freeze_is_frozen(ptz60):
    """SIMULATED: repeat one real frame for 6 s; that run must be reported frozen, not visible."""
    smp = _copy(ptz60)
    for s in range(31, 37):
        smp.grey[s] = smp.grey[30].copy()
    hl = ledger.health("ptz_parking", smp)
    ps = hl["per_second"]
    assert all(ps[s]["reason"] == "frozen" for s in range(30, 37))
    assert ps[25]["visible"] and ps[45]["visible"]


def _fog(x, t, airlight=205.0, rng=None):
    """SIMULATED fog: I = J*t + A*(1-t), plus sensor noise that fog does not remove."""
    y = x.astype(np.float32) * t + airlight * (1 - t)
    if rng is not None:
        y += rng.normal(0, 2.0, x.shape)
    return np.clip(y, 0, 255).astype(np.uint8)


def _lamp(y):
    H, W = y.shape
    yy, xx = np.mgrid[:H, :W]
    return np.clip(y.astype(np.float32) + 220 * np.exp(-((xx - W * 0.7) ** 2 + (yy - H * 0.2) ** 2) / 72.0), 0, 255).astype(np.uint8)


def test_simulated_fog_is_not_blind(ptz60):
    """SIMULATED: fog thickening from t=0.6 to t=0.2 over 50 s on a fixed view must stay visible (not blur/blocked)."""
    smp, rng = _copy(ptz60), np.random.default_rng(1)
    for s in range(smp.seconds):
        smp.grey[s] = _fog(smp.grey[s], 0.6 - 0.4 * s / (smp.seconds - 1), rng=rng)
    hl = ledger.health("ptz_parking", smp)
    assert all(p["visible"] for p in hl["per_second"]), hl["blind"]


def test_simulated_foggy_dawn_with_lamp_is_not_dark(ptz60):
    """SIMULATED: foggy dawn (fog, half exposure, a street lamp on) and dusk (luma ~30) are dim but usable."""
    for gain, fog_t, air in ((0.55, 0.4, 170.0), (0.3, 1.0, 0.0)):
        smp, rng = _copy(ptz60), np.random.default_rng(2)
        for s in range(smp.seconds):
            y = _fog(smp.grey[s], fog_t, air).astype(np.float32) * gain + rng.normal(0, 3.0, smp.grey[s].shape)
            smp.grey[s] = _lamp(np.clip(y, 0, 255).astype(np.uint8))
        hl = ledger.health("ptz_parking", smp)
        assert float(np.mean(hl["stats"]["luma"])) < 80
        assert all(p["visible"] for p in hl["per_second"]), (gain, hl["blind"])


def test_simulated_smear_is_blur(ptz60):
    """SIMULATED: a real smear (Gaussian blur) on 20-30 s is still flagged blur."""
    import cv2
    smp = _copy(ptz60)
    for s in range(20, 30):
        smp.grey[s] = cv2.GaussianBlur(smp.grey[s], (0, 0), 2.0)
    hl = ledger.health("ptz_parking", smp)
    ps = hl["per_second"]
    assert sum(ps[s]["reason"] == "blur" for s in range(20, 30)) >= 8, hl["blind"]
    assert all(ps[s]["visible"] for s in range(5, 15)) and all(ps[s]["visible"] for s in range(35, 50))


def test_simulated_partial_block_is_blocked_not_turned_away(ptz60):
    """SIMULATED: something covers ~55 % of a fixed view for 20 s with no camera motion: 'blocked', never 'turned_away'."""
    smp = _copy(ptz60)
    for s in range(20, 40):
        g = smp.grey[s]
        g[int(g.shape[0] * 0.2):, int(g.shape[1] * 0.1):int(g.shape[1] * 0.8)] = 230   # a white van-sized slab
    hl = ledger.health("ptz_parking", smp)
    rs = [hl["per_second"][s]["reason"] for s in range(20, 40)]
    assert "turned_away" not in rs and "moved" not in rs
    assert sum(r == "blocked" for r in rs) >= 15, hl["blind"]


def test_similarity_same_vs_different():
    """28 -> 533: the same parked car re-tracked after a detector gap; 23 is a white box truck."""
    cs = ledger.CropStore("ptz_parking")
    cs.want(28, ("end",))
    cs.want(23, ("end",))
    cs.want(533, ("start",))
    cs.fetch()
    assert cs.sim(28, 533) >= ledger.SAME_SIM
    assert cs.sim(23, 533) < ledger.SAME_SIM


def _check_contract(led: dict, cam: str) -> None:
    assert set(led) == KEYS
    assert set(led["reason_words"]) == REASONS
    assert led["camera"] == cam
    dur = data.info(cam)["duration"]
    assert isinstance(led["seconds"], int) and led["seconds"] == len(led["per_second"])
    for b in led["blind"]:
        assert set(b) == {"start", "end", "reason", "score"}
        assert b["reason"] in REASONS and 0 <= b["start"] < b["end"] <= dur + 1e-6 and 0 <= b["score"] <= 1
    blind_s = set()
    for p in led["per_second"]:
        assert set(p) == {"t", "visible", "reason"} and isinstance(p["t"], int)
        assert p["visible"] == (p["reason"] is None)
        assert p["reason"] is None or p["reason"] in REASONS
        if not p["visible"]:
            blind_s.add(p["t"])
    covered = sum(b["end"] - b["start"] for b in led["blind"])
    assert abs(covered - len(blind_s)) <= 1.0
    # ONE blind number, and per-day totals that add up to it; day ids/labels come from data.days
    assert abs(led["blind_total_s"] - covered) <= 0.11
    assert abs(sum(led["blind_by_reason"].values()) - led["blind_total_s"]) <= 0.11 * max(1, len(led["blind_by_reason"]))
    days = data.days(cam)
    assert [d["day"] for d in led["blind_by_day"]] == [d["id"] for d in days]
    assert [d["label"] for d in led["blind_by_day"]] == [d["label"] for d in days]
    assert abs(sum(d["blind_s"] for d in led["blind_by_day"]) - led["blind_total_s"]) <= 0.11 * len(days)
    for h in led["home_views"]:
        assert set(h) == {"id", "day", "label", "view", "start", "end", "seen_s"}
        assert h["day"] in {d["id"] for d in days} and h["start"] < h["end"] and h["seen_s"] > 0
    od = data.out_dir(cam)
    for sp in led["spots"]:
        assert set(sp) >= {"spot_id", "box", "spans", "changes"} and len(sp["box"]) == 4
        x1, y1, x2, y2 = sp["box"]
        assert x1 < x2 and y1 < y2
        assert sp["spans"][0]["start"] == 0.0 and abs(sp["spans"][-1]["end"] - dur) < 1.0
        for a, b in zip(sp["spans"], sp["spans"][1:]):
            assert a["end"] == b["start"]
        for s in sp["spans"]:
            assert set(s) == {"start", "end", "state", "object_id", "reason"}
            assert s["state"] in {"present", "gone", "unknown"}
            if s["state"] == "gone":   # never "gone" while blind
                assert not any(s["start"] <= t < s["end"] for t in blind_s), (sp["spot_id"], s)
        for c in sp["changes"]:
            assert {"t", "kind", "before", "after", "same_vehicle", "similarity"} <= set(c)
            assert c["kind"] in {"arrived", "left", "swapped"}
            for k in ("before", "after"):
                if c[k] is not None:
                    assert (od / c[k]).exists() and (od / c[k]).stat().st_size > 500
            if c["kind"] == "arrived":
                assert c["after"] is not None
            if c["kind"] == "left":
                assert c["before"] is not None


def test_cli_contract_car_detection():
    led = ledger.run("car_detection")
    _check_contract(led, "car_detection")
    on_disk = json.loads((data.out_dir("car_detection") / "ledger.json").read_text(encoding="utf-8"))
    assert on_disk["seconds"] == led["seconds"]
    assert (data.out_dir("car_detection") / "ledger_summary.txt").exists()


def test_ptz_ledger_contract_and_truth():
    p = data.out_dir("ptz_parking") / "ledger.json"
    if not p.exists():
        pytest.skip("run `python -m stm.ledger ptz_parking` first (about 2 minutes)")
    led = json.loads(p.read_text(encoding="utf-8"))
    _check_contract(led, "ptz_parking")
    ps = led["per_second"]
    # the PTZ camera turns away from the car park between ~156 s and ~250 s: that must be blind
    assert sum(not ps[s]["visible"] for s in range(160, 236)) >= 70
    assert all(ps[s]["visible"] for s in range(5, 60))
    assert not ps[69]["visible"] and ps[69]["reason"] == "no_signal"     # the recording cut at ~68.2 s
    # the PTZ swing is the camera moving / pointed elsewhere, never "blocked"
    swing = [ps[s]["reason"] for s in range(156, 251)]
    assert "blocked" not in swing and all(swing)
    assert sum(r == "turned_away" for r in swing) >= 40 and sum(r == "moved" for r in swing) >= 20
    assert 90 <= led["blind_total_s"] <= 105
    assert {h["day"] for h in led["home_views"]} == {d["id"] for d in data.days("ptz_parking")}
    assert len(led["spots"]) >= 20
    # a spot is unknown (not gone) while the camera points elsewhere
    for sp in led["spots"]:
        for s in sp["spans"]:
            if s["start"] <= 200 < s["end"]:
                assert s["state"] == "unknown"
