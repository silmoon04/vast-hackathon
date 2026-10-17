"""Privacy-zone setup on a reference frame: Cosmos 2D grounding (may fail) -> classical fallback -> manual polygons."""
from __future__ import annotations
import cv2, numpy as np, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk


def box_poly(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def poly_iou(a, b, shape):
    ma = np.zeros(shape, np.uint8); mb = np.zeros(shape, np.uint8)
    cv2.fillPoly(ma, [np.int32(a)], 1); cv2.fillPoly(mb, [np.int32(b)], 1)
    u = (ma | mb).sum()
    return float((ma & mb).sum() / u) if u else 0.0


def classical_windows(frame, min_area=1000, max_area=40000, max_n=40):
    """Window-like rectangles: locally darker (or locally brighter, lit) blobs with rectangular, window-ish shape."""
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.5, (8, 8)).apply(g)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    sc = w / 1280.0
    cands = []
    for mode in ("dark", "lit"):
        bs = int(max(15, 71 * sc)) | 1
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV if mode == "dark" else cv2.THRESH_BINARY, bs, 4 if mode == "dark" else -25)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(5 * sc)), max(3, int(5 * sc))))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
        cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            x, y, bw, bh = cv2.boundingRect(c)
            a = bw * bh
            if not (min_area * sc * sc <= a <= max_area * sc * sc):
                continue
            ar = bw / bh
            if not (0.35 <= ar <= 3.0):
                continue
            fill = cv2.contourArea(c) / a
            if fill < 0.55:
                continue
            if y + bh > 0.70 * h:        # road / hoardings area: not windows
                continue
            cands.append((fill, mode, (x, y, x + bw, y + bh)))
    cands.sort(key=lambda t: -t[0])
    keep = []
    for fill, mode, b in cands:   # NMS
        if all(poly_iou(box_poly(*b), box_poly(*k[2]), (h, w)) < 0.3 for k in keep):
            keep.append((fill, mode, b))
    return [{"box": list(map(int, b)), "score": round(f, 2), "mode": m, "src": "classical"} for f, m, b in keep[:max_n]]


def cosmos_windows(frame, tries=1):
    """Ask Cosmos (local Edge stand-in for Cosmos3-Reason) for window boxes. Returns (boxes, raw, ok_reason)."""
    h, w = frame.shape[:2]
    img = cv2.convertScaleAbs(frame, alpha=1.5, beta=8)
    prompt = ("Detect every window on the buildings in this image. Output the bounding box of each window in JSON as "
              "[{\"label\":\"window\",\"bbox_2d\":[x1,y1,x2,y2]}] using pixel coordinates of the image.")
    raw = pk.vlm(prompt, [img], max_tokens=300, timeout=60)
    if not raw:
        return [], "", "no answer from the VLM within 60 s (server busy or down)"
    boxes = []
    for m in re.finditer(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", raw):
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        # the model may answer in 0-1000 normalised or pixel coords; accept pixels if they fit the frame
        if max(x2, y2) <= 1000 and (x2 > w or y2 > h):
            x1, x2, y1, y2 = x1 * w / 1000, x2 * w / 1000, y1 * h / 1000, y2 * h / 1000
        if x2 > x1 + 8 and y2 > y1 + 8 and x2 <= w * 1.02 and y2 <= h * 1.02:
            boxes.append({"box": [int(x1), int(y1), int(x2), int(y2)], "src": "cosmos"})
    # sanity: near-duplicate / degenerate list (a typical Edge failure) is rejected
    reason = "ok"
    if len(boxes) == 0:
        reason = "no parsable boxes"
    elif len(boxes) > 60:
        reason = "degenerate repeated list"; boxes = []
    return boxes, raw, reason
