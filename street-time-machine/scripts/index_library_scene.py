"""Run the existing scene builder with native/up-to-720p analysis and bounded CPU use."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from scripts import build_find_scene
from stm import fastindex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cams", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    original_video = fastindex.index_video
    original_cameras = fastindex.index_cameras

    def native_720(path, cam="A", fps=fastindex.FPS, max_s=None, work_w=640):
        cap = cv2.VideoCapture(str(path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        analysis_width = max(2, int(round(width * min(1.0, 720 / max(1, height)))))
        analysis_height = int(round(height * analysis_width / max(1, width)))
        print(f"Indexing camera {cam}: {width}x{height} source, {analysis_width}x{analysis_height} analysis", flush=True)
        result = original_video(path, cam, fps, max_s, work_w=analysis_width)
        print(f"Indexed camera {cam}: {len(result['tracks'])} fragments, {result['seconds']:.1f}s compute", flush=True)
        return result

    def serial_cameras(jobs, fps=fastindex.FPS, max_s=None, workers=None):
        return original_cameras(jobs, fps=fps, max_s=max_s, workers=1)

    def link_source(src, dst, seconds):
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        return None

    fastindex.index_video = native_720
    fastindex.index_cameras = serial_cameras
    build_find_scene.start_transcode = link_source
    build_find_scene.finish_transcode = lambda job, src, dst, log: "native source linked; API serves source"
    sc = build_find_scene.build(args.cams, args.scene, args.out)
    sc["build"]["analysis_max_height"] = 720
    sc["build"]["camera_workers"] = 1
    sc["build"]["identity_provenance"] = "Computed motion and appearance only; no ground truth imported."
    path = Path(args.out) / args.scene / "scene.json"
    path.write_text(json.dumps(sc, indent=1), encoding="utf-8")
    return 1 if sc["build"]["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
