"""Pinch Point occupancy tools over VSS detection sidecars, using only the stdlib.

Preserves the experiment's bottom-centre zone test, rounded median per second,
and literal sustained-threshold semantics. No unique-person count is inferred.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict

VEHICLES = {"car", "truck", "bus", "motorcycle"}


def clock(t):
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def in_polygon(x, y, polygon):
    if not polygon:
        return True
    inside = False
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[i - 1]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-9 and min(ax, bx) - 1e-9 <= x <= max(ax, bx) + 1e-9 and min(ay, by) - 1e-9 <= y <= max(ay, by) + 1e-9:
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def validate_region(value):
    if not value:
        return []
    if not isinstance(value, list) or not 3 <= len(value) <= 12:
        raise ValueError("A region needs 3 to 12 points.")
    result = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each region point needs x and y.")
        x, y = map(float, point)
        if not math.isfinite(x + y) or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Region points must be inside the image.")
        result.append([x, y])
    area = abs(sum(x * result[i - 1][1] - result[i - 1][0] * y for i, (x, y) in enumerate(result))) / 2
    if area < 0.0001:
        raise ValueError("Draw a region with some area.")
    return result


def build(sidecars, duration, region=None, confidence=0.25):
    region = validate_region(region)
    if not math.isfinite(duration) or not 0 < duration <= 3600:
        raise ValueError("Choose a clip between 0 and 3600 seconds long.")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    frames_by_time = {}
    for start, end, side in sidecars:
        fps = float(side.get("fps") or 0)
        if fps <= 0:
            continue
        for frame in side.get("frames", []):
            relative = frame.get("time_sec")
            if relative is None:
                relative = float(frame.get("frame_index", 0)) / fps
            t = float(start) + float(relative)
            if not math.isfinite(t) or t < start or t >= min(float(end), duration) + 0.001:
                continue
            shape = frame.get("shape") or side.get("video_shape")
            if not shape or len(shape) < 2:
                continue
            height, width = map(float, shape[:2])
            if height <= 0 or width <= 0:
                continue
            counts = defaultdict(int)
            boxes = []
            for detection in frame.get("detections", []):
                label = str(detection.get("label") or detection.get("name") or "").lower()
                box = detection.get("bbox") or detection.get("xyxy")
                conf = float(detection.get("confidence", detection.get("conf", 0)))
                if not box or len(box) != 4 or conf < confidence or label not in VEHICLES | {"person", "bicycle"}:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2, conf)) or x2 <= x1 or y2 <= y1:
                    continue
                inside = in_polygon((x1 + x2) / (2 * width), min(y2, height - 1) / height, region)
                if inside:
                    counts[label] += 1
                    if label in VEHICLES:
                        counts["vehicle"] += 1
                boxes.append({"label": label, "confidence": round(conf, 3), "box": [round(x1 / width, 5), round(y1 / height, 5), round(x2 / width, 5), round(y2 / height, 5)], "in_region": inside})
            frames_by_time[round(t, 4)] = {"t": round(t, 4), "counts": dict(counts), "boxes": boxes}
    frames = sorted(frames_by_time.values(), key=lambda f: f["t"])
    if not frames:
        raise ValueError("This clip has no usable frame detections. Choose another catalogue result.")
    buckets = defaultdict(list)
    for frame in frames:
        buckets[int(frame["t"])].append(frame)
    labels = ["person", "vehicle", "car", "truck", "bus", "motorcycle", "bicycle"]
    series = []
    for sec in range(math.ceil(duration)):
        observed = buckets.get(sec, [])
        counts = {label: int(math.floor(statistics.median(f["counts"].get(label, 0) for f in observed) + .5)) for label in labels} if observed else None
        series.append({"t": sec, "end": min(sec + 1, duration), "counts": counts, "sampled_frames": len(observed)})
    covered = sum(row["end"] - row["t"] for row in series if row["counts"] is not None)
    return {"duration": round(duration, 3), "region": region, "confidence": confidence,
            "frame_count": len(frames), "covered_seconds": round(covered, 3),
            "coverage_pct": round(100 * covered / duration, 1), "series": series, "frames": frames,
            "method": "Pinch Point: box bottom-centres inside the region; rounded median count per second.",
            "limit": "Counts are simultaneous detector observations, not unique people or vehicles. Missing detection data stays unknown."}


def threshold_events(series, label, threshold, sustain):
    if threshold < 1 or sustain <= 0:
        raise ValueError("Use a positive count and duration.")
    events, run = [], []

    def finish():
        if run and run[-1]["end"] - run[0]["t"] >= sustain:
            peak = max(run, key=lambda row: row["counts"].get(label, 0))
            events.append({"start": run[0]["t"], "end": run[-1]["end"], "fire_at": run[0]["t"] + sustain,
                           "peak": peak["counts"].get(label, 0), "peak_t": peak["t"]})
    for row in series:
        if row["counts"] is not None and row["counts"].get(label, 0) >= threshold:
            if run and abs(run[-1]["end"] - row["t"]) > 0.001:
                finish()
                run = []
            run.append(row)
        else:
            finish()
            run = []
    finish()
    return events


def answer(analysis, question, mode="people"):
    q = question.lower().strip()
    label = "vehicle" if mode == "vehicles" else "person"
    names = {"person": "people", "vehicle": "vehicles", "car": "cars", "truck": "trucks", "bus": "buses", "bicycle": "bicycles", "motorcycle": "motorcycles"}
    for word, target in [("people", "person"), ("pedestrian", "person"), ("person", "person"), ("vehicle", "vehicle"), ("cars", "car"), ("car", "car"), ("truck", "truck"), ("buses", "bus"), ("bus", "bus"), ("bicycle", "bicycle"), ("cyclist", "bicycle"), ("motorcycle", "motorcycle")]:
        if re.search(r"\b" + word + r"s?\b", q):
            label = target
            break
    noun = names[label]
    observed = [row for row in analysis["series"] if row["counts"] is not None]
    peak = max(observed, key=lambda row: row["counts"].get(label, 0))
    n = peak["counts"].get(label, 0)
    evidence = [{"time": peak["t"], "count": n, "label": label}]
    result = {"question": question, "label": label, "evidence": evidence, "engine": "computed", "intent": "peak"}
    if re.search(r"\b(unique|different|individual|altogether)\b|\bin total\b", q):
        text = f"I can count {noun} visible together, but cannot count unique identities in this feature. The highest one-second median is {n} at {clock(peak['t'])}."
        result["intent"] = "unique_unavailable"
    elif re.search(r"\b(coverage|analysed|analyzed|missing|complete|gaps)\b", q):
        text = f"Detection data covers {analysis['covered_seconds']:g} of {analysis['duration']:g} seconds ({analysis['coverage_pct']:g}%), across {analysis['frame_count']} frames. Seconds without usable detections are unknown."
        result["intent"] = "coverage"
    elif re.search(r"\b(at least|above|over|threshold|sustain|stayed|stay|for \d)\b", q):
        count_match = re.search(r"(?:at least|above|over|threshold(?: of)?)\s+(\d+)", q) or re.search(r"\b(\d+)\s+(?:people|persons?|pedestrians?|vehicles?|cars?|trucks?|buses?|bicycles?)", q)
        threshold = int(count_match[1]) if count_match else max(1, n // 2)
        if re.search(r"\b(over|above)\b", q) and not re.search(r"at least", q):
            threshold += 1
        duration_match = re.search(r"for\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", q)
        sustain = float(duration_match[1]) if duration_match else 3.0
        events = threshold_events(analysis["series"], label, threshold, sustain)
        spans = ", ".join(f"{clock(e['start'])}-{clock(e['end'])}" for e in events[:5])
        text = f"{len(events)} interval{'s' if len(events) != 1 else ''} met at least {threshold} {noun} for {sustain:g} seconds."
        if events:
            text += f" {spans}. Every observed second in each interval meets the threshold."
        else:
            text += f" The highest one-second median was {n} at {clock(peak['t'])}."
        result.update(intent="threshold", events=events, threshold=threshold, sustain=sustain)
        result["evidence"] = [{"time": e["peak_t"], "count": e["peak"], "label": label} for e in events[:5]] or evidence
    elif re.search(r"\b(what|which)\s+(?:kinds? of\s+|types? of\s+)?(?:vehicles?|objects?)", q):
        values = {kind: max(row["counts"].get(kind, 0) for row in observed) for kind in ["car", "truck", "bus", "motorcycle", "bicycle"]}
        parts = [f"{names[k]} (peak median {v})" for k, v in values.items() if v]
        text = "Detected in the selected region: " + (", ".join(parts) if parts else "no vehicles at the current confidence setting") + "."
        result.update(intent="classes", classes=values)
    elif re.search(r"\b(between|from)\s+\d", q):
        m = re.search(r"(?:between|from)\s+(\d+(?::\d+(?:\.\d+)?)?)\s*(?:seconds?|secs?|s)?\s*(?:and|to|-)\s*(\d+(?::\d+(?:\.\d+)?)?)", q)
        if not m:
            text = "Use a range such as 'How many people between 5 and 15 seconds?'"
            result["intent"] = "help"
        else:
            def seconds(value):
                return sum(float(v) * (60 ** i) for i, v in enumerate(reversed(value.split(":"))))
            a, b = seconds(m[1]), seconds(m[2])
            subset = [r for r in observed if a <= r["t"] < b]
            if not subset:
                text = "There is no analysed data in that time range."
            else:
                p = max(subset, key=lambda r: r["counts"].get(label, 0))
                text = f"Between {clock(a)} and {clock(b)}, the highest one-second median was {p['counts'].get(label, 0)} {noun} at {clock(p['t'])}."
                result["evidence"] = [{"time": p["t"], "count": p["counts"].get(label, 0), "label": label}]
            result["intent"] = "range"
    elif re.search(r"\b(busiest|peak|maximum|most|many|count|visible|present|quietest|least|empty)\b", q):
        if re.search(r"\b(quietest|least|empty)\b", q):
            peak = min(observed, key=lambda row: row["counts"].get(label, 0))
            n = peak["counts"].get(label, 0)
            text = f"The lowest one-second median is {n} {noun} at {clock(peak['t'])}."
            result["evidence"] = [{"time": peak["t"], "count": n, "label": label}]
        else:
            frame_peak = max(analysis["frames"], key=lambda f: f["counts"].get(label, 0))
            text = f"The busiest second is {clock(peak['t'])}, with a median of {n} {noun} visible in the selected region. The highest individual frame has {frame_peak['counts'].get(label, 0)} at {clock(frame_peak['t'])}."
    else:
        text = "I can answer questions about visible people or vehicles, busiest times, sustained counts, time ranges and data coverage. Try 'When was it busiest?' or 'Were at least 5 people visible for 3 seconds?'"
        result.update(intent="help", evidence=[])
    result["answer"] = text
    return result
