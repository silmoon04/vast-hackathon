"""Fast invariants for ON THE LINE (< 60 s). Needs run.py --all to have produced out/ and results.json.
Run: .venv/Scripts/python.exe prototypes/08_on_the_line/test_smoke.py   (or pytest)"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import protokit as pk  # noqa: E402

import engine  # noqa: E402
import qa  # noqa: E402
import sim  # noqa: E402

CLIPS = ["person_bicycle_car_detection", "rochester_night", "retail_cctv"]
RES = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
SCRIPTS = json.loads((HERE / "scripts.json").read_text(encoding="utf-8"))


def test_outputs_exist_and_sized():
    for c in CLIPS:
        d = HERE / "out" / c
        for f in ("report.html", "hero.jpg", "demo.mp4"):
            assert (d / f).exists() and (d / f).stat().st_size > 5_000, f"{c}/{f} missing"
        im = cv2.imread(str(d / "hero.jpg"))
        assert im is not None and im.shape[1] <= 1280 and im.shape[0] >= 600, f"{c} hero size {None if im is None else im.shape}"
        html = (d / "report.html").read_text(encoding="utf-8")
        assert "hero.jpg" in html and "CC" in html or "Public" in html or "Intel" in html or "licen" in html.lower()


def test_five_scored_questions_per_clip_and_latency_under_1s():
    for c in CLIPS:
        r = RES["clips"][c]
        assert r["answers_scored"] == 5, c
        assert r["latency_ms"]["max"] < 1000, (c, r["latency_ms"]["max"])
        assert len(r["questions"]) >= 5


def test_kill_test_matches_numbers():
    kt = RES["kill_test"]
    good = [c for c in CLIPS if RES["clips"][c]["answers_right"] >= 4]
    assert kt["clips_with_ge_4_of_5"] == good
    assert kt["passed"] == (len(good) >= 2 and kt["max_latency_ms"] < 1000)


def test_no_future_data_in_answers():
    """results.json: every question re-run on data truncated at its own time gave identical facts."""
    for c in CLIPS:
        r = RES["clips"][c]
        assert r["causality"], f"{c}: causality check not run"
        assert all(x["identical"] for x in r["causality"]), (c, r["causality"])


def test_live_replay_matches_stored_answers_on_one_clip():
    """Fresh no-VLM replay of the car-park clip reproduces the stored right/wrong verdicts and no answer uses a
    frame later than its question time."""
    name = "person_bicycle_car_detection"
    t0 = time.time()
    res = sim.simulate(name, SCRIPTS[name], use_vlm=False)
    assert time.time() - t0 < 40
    stored = {q["id"]: q["correct"] for q in RES["clips"][name]["questions"]}
    for s in res.scored:
        assert s["correct"] == stored[s["id"]], s["id"]
        assert s["latency_ms"] < 1000
    for c in res.cards:  # evidence crops were taken at or before the question time
        for ev in c.evidence:
            assert ev.t <= c.t + 1.0 / 12 + 1e-6, (c.question, ev.t, c.t)


def test_truncation_gives_identical_facts():
    name = "person_bicycle_car_detection"
    full = sim.simulate(name, SCRIPTS[name], use_vlm=False, end=11.0)
    part = sim.simulate(name, SCRIPTS[name], use_vlm=False, only=1)
    a = next(s for s in full.scored if s["i"] == 1)["facts"]
    b = next(x for x in part.scored if x["i"] == 1)["facts"]
    assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)


def test_entity_invariants():
    name = "rochester_night"
    res = sim.simulate(name, SCRIPTS[name], use_vlm=False)
    st = res.state
    seen_tids: set[int] = set()
    for e in st.entities.values():
        assert e.first_t <= e.last_t
        assert not (set(e.raw_tids) & seen_tids), "raw tracker id owned by two entities"
        seen_tids |= set(e.raw_tids)
        assert all(st.alarm_t - 1e-6 <= p[0] <= st.now + 1e-6 for p in e.pts)
    assert any(e.carried for e in st.entities.values()), "car on the recovery truck should be flagged as cargo"
    assert sum(1 for e in st.confirmed("vehicle")) == 4


def test_parser_intents():
    cases = {
        "How many people can you see right now?": ({"count"}, "person"),
        "Which way did he go?": ({"direction"}, None),
        "What was he wearing?": ({"appearance"}, None),
        "Is the car still there?": ({"still"}, "vehicle"),
        "When did the first vehicle arrive?": ({"arrival"}, "vehicle"),
        "How many cars have come in since the alarm?": ({"total"}, "vehicle"),
    }
    for text, (intents, group) in cases.items():
        a = qa.parse(text, None)
        assert a is not None and intents <= set(a.intents), (text, a)
        if group:
            assert a.group == group, (text, a.group)
    assert qa.parse("lovely weather today", None) is None


def test_colour_names():
    assert engine.colour_name((250, 250, 250)) == "white"
    assert engine.colour_name((10, 10, 10)) == "black"
    assert engine.colour_name((200, 40, 20)) == "blue"
    assert engine.colour_name((30, 30, 220)) == "red"
    assert engine.colour_name((120, 120, 120)) == "grey"


def test_incident_log_has_alarm_and_ordered_times():
    for c in CLIPS:
        txt = RES["clips"][c]["incident_log"].splitlines()
        assert txt[0].startswith("INCIDENT LOG") and "Alarm received" in txt[1]
        ts = [l.split()[0] for l in txt[1:] if l[:2].isdigit()]
        assert ts == sorted(ts), c


def test_parser_paraphrases():
    """Phrasings police actually use. The first 6 are the ones the reviewer saw fail; the rest were written afterwards."""
    cases = [
        ("Where'd they go?", {"direction"}, None),
        ("What's he got on?", {"appearance"}, None),
        ("How many blokes?", {"count"}, "person"),
        ("The man in the red jacket, where is he?", {"direction"}, "person"),
        ("What've they got on?", {"appearance"}, None),
        ("Which way did the motor head off?", {"direction"}, "vehicle"),
        ("Any idea which direction the van went?", {"direction"}, "vehicle"),
        ("Is anybody still about?", {"count"}, "person"),
        ("Has the red one gone yet?", {"still"}, None),
        ("When did the last one turn up?", {"arrival"}, None),
        ("How many vehicles have there been all told?", {"total"}, "vehicle"),
        ("Can you give me a description of the lad?", {"appearance"}, "person"),
        ("What colour was the car, and what time did it show up?", {"appearance", "arrival"}, "vehicle"),
        ("How many of them are there?", {"count"}, None),
    ]
    for text, intents, group in cases:
        a = qa.parse(text, None)
        assert a is not None and intents <= set(a.intents), (text, a)
        if group:
            assert a.group == group, (text, a.group)
    a = qa.parse("The man in the red jacket, where is he?", None)
    assert a.selector == "attr" and a.attr.part == "top" and "red" in a.attr.cols
    a = qa.parse("What's the lad in the black top got on?", None)
    assert a.selector == "attr" and a.attr.cols >= {"black"}
    assert qa.parse("lovely weather today", None) is None and not qa.looks_like_question("lovely weather today")
    assert qa.looks_like_question("Are they armed")


def test_unparsed_question_gets_a_fallback_card_not_silence():
    name = "person_bicycle_car_detection"
    st = sim.simulate(name, SCRIPTS[name], use_vlm=False, end=9.0).state
    card, _ = qa.answer(st, "Are they armed?", None)
    assert card is not None and card.intents == ["summary"] and "Didn't catch" in card.headline and card.caveats
    assert "summary" in card.facts and card.latency_ms < 1000
    none_card, _ = qa.answer(st, "lovely weather today", None)
    assert none_card is None            # small talk stays silent


def test_attribute_selector_finds_the_red_car_on_a_held_out_clip():
    name = "car_detection"
    res = sim.simulate(name, SCRIPTS[name], use_vlm=False, only=3)
    c4 = next(s for s in res.scored if s["id"] == "C4")
    assert c4["correct"], c4
    st = res.state
    red = [e for e in st.confirmed("vehicle") if st.colours(e).get("body") == "red"]
    assert len(red) == 1 and c4["facts"]["still"]["eid"] == red[0].eid


def test_cosmos_conflict_check_and_clothing_words():
    # the retail S2 sentence the reviewer flagged: a child in a blue top for a grey-sweater adult
    bad = "A child with light-colored hair wearing a blue top and trousers, a light-colored hat"
    assert qa.cosmos_conflict(bad, "person", {"top": "silver", "bottom": "navy"}) == "Cosmos guessed an age"
    assert qa.cosmos_conflict("A person wearing a blue top and a striped skirt", "person", {"top": "silver"}).startswith("Cosmos says blue top")
    assert qa.cosmos_conflict("A person wearing a light grey sweater and blue jeans", "person", {"top": "silver", "bottom": "navy"}) is None
    assert qa.cosmos_conflict("White compact hatchback", "vehicle", {"body": "white"}) is None
    lines, cav, facts = qa.cosmos_lines({"desc": bad, "colours": {"top": "silver"}, "desc_pending": False}, "person")
    assert cav == ["descriptions disagree"] and "Cosmos:" not in lines[0] and "desc" not in facts
    assert engine.display_colour("person", "silver") == "light grey" and engine.display_colour("vehicle", "silver") == "silver"
    ret = RES["clips"]["retail_cctv"]
    s2 = next(q for q in ret["questions"] if q["id"] == "S2")
    assert "silver" not in s2["headline"] and "light grey" in s2["headline"]


def test_held_out_clips_recorded_with_paraphrased_questions():
    ho = RES["held_out"]
    assert set(ho["clips"]) == {"one_by_one_person_detection", "car_detection", "people_detection"}
    assert ho["scored"] == sum(RES["clips"][c]["answers_scored"] for c in ho["clips"])
    assert ho["right"] == sum(RES["clips"][c]["answers_right"] for c in ho["clips"])
    for c in ("car_detection", "people_detection"):
        assert SCRIPTS[c].get("paraphrase") and RES["clips"][c]["causality_all_identical"]


def test_hero_badge_sits_in_the_header_not_over_the_frame():
    for c in CLIPS:
        im = cv2.imread(str(HERE / "out" / c / "hero.jpg"))
        head = im[8:54, 300:800]
        assert ((head[:, :, 1] > 150) & (head[:, :, 2] < 120)).sum() > 200, f"{c}: no green verdict badge in the header strip"
        video_top = im[80:200, 440:740]       # where the old tilted badge used to sit
        assert not ((video_top[:, :, 1] > 150) & (video_top[:, :, 2] < 90) & (video_top[:, :, 0] < 140)).sum() > 1500, c


if __name__ == "__main__":
    t0 = time.time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(fns)} smoke checks passed in {time.time() - t0:.1f} s")
