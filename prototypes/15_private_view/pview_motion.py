"""Camera-motion estimation vs a reference frame (feature matching + RANSAC homography).
H maps REFERENCE pixel coords -> CURRENT frame pixel coords (full-resolution coords)."""
from __future__ import annotations
import cv2, numpy as np
cv2.setNumThreads(1)   # many workers share this machine: thread oversubscription made ORB 17x slower
from dataclasses import dataclass, field

WORK_W = 640  # features are computed at this width, H is rescaled to full-res coords


def _prep(frame, work_w=WORK_W):
    s = work_w / frame.shape[1] if frame.shape[1] > work_w else 1.0
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if s != 1.0:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)).apply(g)
    return g, s


class Features:
    def __init__(self, kind="sift"):
        self.kind = kind
        if kind == "orb":
            self.det = cv2.ORB_create(1500, scaleFactor=1.2, nlevels=10, fastThreshold=10)
            self.norm = cv2.NORM_HAMMING
        else:
            self.det = cv2.SIFT_create(1000, contrastThreshold=0.02)
            self.norm = cv2.NORM_L2
        self.bf = cv2.BFMatcher(self.norm)

    def extract(self, frame, mask=None):
        g, s = _prep(frame)
        m = None
        if mask is not None:
            m = cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        kp, des = self.det.detectAndCompute(g, m)
        pts = np.float32([k.pt for k in kp]) / s if kp else np.zeros((0, 2), np.float32)
        return {"pts": pts, "des": des, "n": len(kp)}

    def match(self, a, b, ratio=0.78, ransac=3.0):
        """Homography a->b in full-res coords. Returns (H or None, inliers, n_good)."""
        if a["des"] is None or b["des"] is None or a["n"] < 8 or b["n"] < 8:
            return None, 0, 0
        raw = self.bf.knnMatch(a["des"], b["des"], k=2)
        good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < ratio * n.distance]
        if len(good) < 12:
            return None, 0, len(good)
        src = a["pts"][[m.queryIdx for m in good]]
        dst = b["pts"][[m.trainIdx for m in good]]
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac, maxIters=3000, confidence=0.995)
        if H is None:
            return None, 0, len(good)
        return H, int(inl.sum()), len(good)


def plausible(H, w, h):
    if H is None or not np.all(np.isfinite(H)):
        return False
    A = H[:2, :2]
    det = np.linalg.det(A)
    if det < 0.15 or det > 7.0:
        return False
    if abs(H[2, 0]) * w > 0.8 or abs(H[2, 1]) * h > 0.8:  # strong perspective: unreliable
        return False
    c = np.array([[w / 2, h / 2, 1.0]]).T
    p = H @ c
    if abs(p[2, 0]) < 1e-6:
        return False
    return True


@dataclass
class MotionTracker:
    """H(ref -> current) per frame. Direct match to the reference first (no drift); if that fails, chain through the
    last good frame / recent keyframes; if everything fails the view is 'lost' (not recognised)."""
    ref_frame: np.ndarray
    kind: str = "sift"
    min_inliers: int = 50
    feat: Features = None
    ref: dict = None
    last: dict = None          # last good: {"f": features, "H": H_ref->that frame}
    bank: list = field(default_factory=list)
    lost_run: int = 0
    stats: dict = field(default_factory=lambda: {"ref": 0, "chain": 0, "lost": 0})

    def __post_init__(self):
        self.feat = Features(self.kind)
        self.h, self.w = self.ref_frame.shape[:2]
        self.ref = self.feat.extract(self.ref_frame)
        self.last = {"f": self.ref, "H": np.eye(3)}
        self._hi = None

    def _hires(self, frame):
        """Accurate re-anchor against the reference: full-resolution SIFT (slow, only used when the cheap match is weak)."""
        global WORK_W
        old = WORK_W
        WORK_W = 1280
        try:
            if self._hi is None:
                self._hi = Features("sift")
                self._hi.det = cv2.SIFT_create(3000, contrastThreshold=0.015)
                self._hi_ref = self._hi.extract(self.ref_frame)
            cur = self._hi.extract(frame)
            Hr, inl, ng = self._hi.match(self._hi_ref, cur, ratio=0.85)
        finally:
            WORK_W = old
        if Hr is not None and inl >= 90 and plausible(Hr, self.w, self.h):
            return {"H": Hr, "inliers": inl, "src": "ref-hires"}
        return None

    def update(self, frame):
        """Returns dict: H (ref->cur) or None, inliers, src ('ref' | 'last' | 'bank' | 'lost')."""
        cur = self.feat.extract(frame)
        best = None
        cands = [("ref", self.ref, np.eye(3))]
        if self.lost_run < 40:                      # a long loss: stop chaining from stale frames, only try the reference
            cands.append(("last", self.last["f"], self.last["H"]))
            for kf in self.bank[::-1][:2]:
                cands.append(("bank", kf["f"], kf["H"]))
        for name, f, Hk in cands:
            Hr, inl, ng = self.feat.match(f, cur)
            if Hr is None or inl < self.min_inliers:
                continue
            H = Hr @ Hk
            if not plausible(H, self.w, self.h):
                continue
            if best is None or inl > best["inliers"]:
                best = {"H": H, "inliers": inl, "src": name}
            if name == "ref" and inl >= 90:        # strong direct match to the reference: stop
                break
        if best is not None and best["src"] == "ref" and best["inliers"] < 150:
            hb = self._hires(frame)            # weak direct match: re-anchor at full resolution
            if hb is not None:
                best = hb
        if best is None and self.lost_run % 4 == 0:
            best = self._hires(frame)           # lost: occasionally try the slow accurate matcher
        if best is None:
            self.lost_run += 1
            self.stats["lost"] += 1
            return {"H": None, "inliers": 0, "src": "lost", "n_feat": cur["n"]}
        self.lost_run = 0
        self.stats["ref" if best["src"].startswith("ref") else "chain"] += 1
        self.last = {"f": cur, "H": best["H"]}
        if (not self.bank or np.abs(self.bank[-1]["H"] - best["H"]).max() > 0.15) and best["inliers"] > 50:
            self.bank.append(self.last)
            self.bank = self.bank[-12:]
        best["n_feat"] = cur["n"]
        return best


# ---------------------------------------------------------------------------------------------------------------
# Independent second estimator for CROSS-CHECKS ONLY (not used by the monitor): ECC, an intensity-based alignment with no
# feature matching. It shares no code and no failure mode with SIFT+RANSAC, so agreement between the two on real footage is
# evidence that is not circular. It only copes with moderate motion (it starts from the previous solution) and needs the
# image region used to be rigid (moving cars are masked out by `top_frac`).
class EccTracker:
    """H(ref -> current) by findTransformECC (homography), chained from the previous sample's solution."""

    def __init__(self, ref_frame, top_frac: float = 0.62, scale: float = 0.5):
        self.S = scale
        self.Sm = np.diag([scale, scale, 1.0])
        self.Si = np.linalg.inv(self.Sm)
        self.rg = self._prep(ref_frame)
        self.mask = np.zeros(self.rg.shape, np.uint8)
        self.mask[: int(top_frac * self.rg.shape[0])] = 1
        self.prev = np.eye(3)
        self.last_cc = 1.0

    def _prep(self, f):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=self.S, fy=self.S, interpolation=cv2.INTER_AREA)
        g = cv2.createCLAHE(3.0, (6, 6)).apply(g)
        return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)

    def update(self, frame):
        W = (self.Sm @ self.prev @ self.Si).astype(np.float32)
        try:
            cc, W = cv2.findTransformECC(self.rg, self._prep(frame), W, cv2.MOTION_HOMOGRAPHY,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6), self.mask, 5)
        except cv2.error:
            return None
        self.last_cc = float(cc)
        self.prev = self.Si @ W.astype(np.float64) @ self.Sm
        return self.prev.copy()
