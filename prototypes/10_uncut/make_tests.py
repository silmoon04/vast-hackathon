"""Build the doctored test clips with ffmpeg. Ground truth is declared HERE, before any matching code runs.

source times are in source-video seconds; 'removed' = (source_time_where_gap_starts, gap_seconds).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402

HERE = Path(__file__).resolve().parent
TESTS = HERE / "tests"
F = pk.FFMPEG

# ground truth per test clip
GT = {
    "t1_lisbon_crop80": {"verdict": "MATCH", "source": "lisbon_parking", "src_start": 60.0, "src_end": 75.0, "speed": 1.0,
                         "mirrored": False, "crop": 0.8, "removed": [], "foreign": [],
                         "edit": "centre crop 80% rescaled to 640x480, x264 crf 32"},
    "t2_retail_mirror": {"verdict": "MATCH", "source": "retail_cctv", "src_start": 20.0, "src_end": 35.0, "speed": 1.0,
                         "mirrored": True, "crop": 1.0, "removed": [], "foreign": [], "edit": "hflip, x264 crf 28"},
    "t3_rochester_speed_cut": {"verdict": "MATCH", "source": "rochester_night", "src_start": 10.0, "src_end": 25.0,
                               "speed": 1.25, "mirrored": False, "crop": 1.0, "removed": [(17.5, 3.0)], "foreign": [],
                               "edit": "source 10-17.5 + 20.5-25 (3 s removed), played at 1.25x, crf 28"},
    "t4_splice_ptz_sitdown": {"verdict": "PARTIAL", "source": "ptz_parking", "src_start": 100.0, "src_end": 105.0, "speed": 1.0,
                              "mirrored": False, "crop": 1.0, "removed": [], "foreign": [(5.0, 10.0)],
                              "edit": "5 s ptz_parking (100-105 s) + 5 s sit_down (not in archive), letterboxed to 1280x720"},
    # the spec's demo moment: crop AND cut in one clip. Added after the first review; written down before it was run.
    "t6_lisbon_crop_cut": {"verdict": "MATCH", "source": "lisbon_parking", "src_start": 60.0, "src_end": 78.0, "speed": 1.0,
                           "mirrored": False, "crop": 0.8, "removed": [(67.0, 3.0)], "foreign": [],
                           "edit": "lisbon 60-67 s + 70-78 s (3 s removed at 67 s), centre crop 80% rescaled to 640x480, crf 32"},
    "t5_skate_fall": {"verdict": "NOT IN ARCHIVE", "source": None, "edit": "skate_fall.mp4 entirely, re-encoded crf 28"},
    # extras (not part of the 5-clip kill test)
    "x6_lisbon_holdout": {"verdict": "NOT IN ARCHIVE", "source": None,
                          "edit": "lisbon_parking 300-315 s, untouched, but that window is held OUT of the archive"},
    "x7_retail_control": {"verdict": "MATCH", "source": "retail_cctv", "src_start": 50.0, "src_end": 65.0, "speed": 1.0,
                          "mirrored": False, "crop": 1.0, "removed": [], "foreign": [], "edit": "control: plain cut, crf 23"},
    # stress extras (honest limits; not part of the 5-clip kill test)
    "x8_retail_combo": {"verdict": "MATCH", "source": "retail_cctv", "src_start": 70.0, "src_end": 92.5, "speed": 1.5,
                        "mirrored": True, "crop": 0.7, "variant": "0.7M", "removed": [], "foreign": [],
                        "edit": "mirrored + centre crop 70% + 1.5x speed (15 s query = 22.5 s of source), crf 30"},
    "x9_dashcam_offcentre": {"verdict": "MATCH", "source": "dashcam_urban", "src_start": 30.0, "src_end": 45.0, "speed": 1.0,
                             "mirrored": False, "crop": "off-centre 70% (right side)", "variant": "0.7R", "removed": [], "foreign": [],
                             "edit": "crop 70% from the RIGHT side (not centred), rescaled; archive only stores CENTRE crops"},
    "x10_yonge_lowres_grade": {"verdict": "MATCH", "source": "yonge_dundas", "src_start": 10.0, "src_end": 25.0, "speed": 1.0,
                               "mirrored": False, "crop": 1.0, "removed": [], "foreign": [],
                               "edit": "320x180, brightness +0.12, contrast 1.3, saturation 0, x264 crf 36"},
    "x16_lisbon_cut_in_static": {"verdict": "MATCH", "source": "lisbon_parking", "src_start": 60.0, "src_end": 78.0, "speed": 1.0,
                                 "mirrored": False, "crop": 0.8, "removed": [(66.0, 3.0)], "foreign": [],
                                 "edit": "lisbon 60-66 s + 69-78 s: the 3 s cut starts 1 s before the end of the static title card (card ~64-67 s), "
                                         "so the frames just before the cut are static; crop 80%, crf 32"},
    "x13_dashcam_916_caption": {"verdict": "MATCH", "source": "dashcam_urban", "src_start": 60.0, "src_end": 72.0, "speed": 1.0,
                                "mirrored": False, "crop": "9:16 portrait, right edge", "variant": "p916R", "removed": [], "foreign": [],
                                "edit": "16:9 -> 9:16 portrait strip from the RIGHT edge, 406x720, burned-in white caption + @handle, crf 30"},
    "x14_retail_tiktok_bars": {"verdict": "MATCH", "source": "retail_cctv", "src_start": 40.0, "src_end": 52.0, "speed": 1.0,
                               "mirrored": False, "crop": 1.0, "removed": [], "foreign": [],
                               "edit": "whole 16:9 frame letterboxed into a 720x1280 portrait canvas (black bars top+bottom), burned-in caption, crf 30"},
    "x15_yonge_916_offgrid": {"verdict": "MATCH", "source": "yonge_dundas", "src_start": 20.0, "src_end": 32.0, "speed": 1.0,
                              "mirrored": False, "crop": "9:16 portrait at 75% across (between the indexed centre and right)", "removed": [],
                              "foreign": [], "edit": "9:16 portrait strip positioned 75% of the way to the right edge (NOT an indexed position), crf 30"},
    "x12_lisbon_titlecard": {"verdict": "MATCH", "source": "lisbon_parking", "src_start": 66.0, "src_end": 69.0, "speed": 1.0,
                             "mirrored": False, "crop": 1.0, "removed": [], "foreign": [],
                             "edit": "3 s of the static title card inside lisbon_parking, untouched"},
}
EXTRAS = ["t6_lisbon_crop_cut"] + [k for k in GT if k.startswith("x")]
KILL = ["t1_lisbon_crop80", "t2_retail_mirror", "t3_rochester_speed_cut", "t4_splice_ptz_sitdown", "t5_skate_fall"]
HERO = "t6_lisbon_crop_cut"      # opening slide of the demo: crop + cut in one clip


def run(*args: str) -> None:
    subprocess.run([F, "-y", "-v", "error", *args], check=True)


def src(n: str) -> str:
    return str(pk.video_path(n))


def make(force: bool = False) -> dict[str, Path]:
    TESTS.mkdir(exist_ok=True)
    out = {k: TESTS / f"{k}.mp4" for k in GT}
    enc = lambda crf: ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an"]  # noqa: E731
    if force or not out["t1_lisbon_crop80"].exists():
        run("-ss", "60", "-i", src("lisbon_parking"), "-t", "15", "-vf", "crop=iw*0.8:ih*0.8,scale=640:480", *enc(32),
            str(out["t1_lisbon_crop80"]))
    if force or not out["t2_retail_mirror"].exists():
        run("-ss", "20", "-i", src("retail_cctv"), "-t", "15", "-vf", "hflip", *enc(28), str(out["t2_retail_mirror"]))
    if force or not out["t3_rochester_speed_cut"].exists():
        fc = ("[0:v]trim=start=10:end=17.5,setpts=PTS-STARTPTS[a];[0:v]trim=start=20.5:end=25,setpts=PTS-STARTPTS[b];"
              "[a][b]concat=n=2:v=1:a=0,setpts=PTS/1.25[o]")
        run("-i", src("rochester_night"), "-filter_complex", fc, "-map", "[o]", *enc(28), str(out["t3_rochester_speed_cut"]))
    if force or not out["t4_splice_ptz_sitdown"].exists():
        fc = ("[0:v]trim=start=100:end=105,setpts=PTS-STARTPTS,fps=25,scale=1280:720[a];"
              "[1:v]trim=start=0:end=5,setpts=PTS-STARTPTS,fps=25,scale=-2:720,pad=1280:720:(ow-iw)/2:0:black[b];"
              "[a][b]concat=n=2:v=1:a=0[o]")
        run("-i", src("ptz_parking"), "-i", src("sit_down"), "-filter_complex", fc, "-map", "[o]", *enc(28),
            str(out["t4_splice_ptz_sitdown"]))
    if force or not out["t5_skate_fall"].exists():
        run("-i", src("skate_fall"), *enc(28), str(out["t5_skate_fall"]))
    if force or not out["t6_lisbon_crop_cut"].exists():
        fc = ("[0:v]trim=start=60:end=67,setpts=PTS-STARTPTS[a];[0:v]trim=start=70:end=78,setpts=PTS-STARTPTS[b];"
              "[a][b]concat=n=2:v=1:a=0,crop=iw*0.8:ih*0.8,scale=640:480[o]")
        run("-i", src("lisbon_parking"), "-filter_complex", fc, "-map", "[o]", *enc(32), str(out["t6_lisbon_crop_cut"]))
    if force or not out["x16_lisbon_cut_in_static"].exists():
        fc = ("[0:v]trim=start=60:end=66,setpts=PTS-STARTPTS[a];[0:v]trim=start=69:end=78,setpts=PTS-STARTPTS[b];"
              "[a][b]concat=n=2:v=1:a=0,crop=iw*0.8:ih*0.8,scale=640:480[o]")
        run("-i", src("lisbon_parking"), "-filter_complex", fc, "-map", "[o]", *enc(32), str(out["x16_lisbon_cut_in_static"]))
    FONT = r"fontfile='C\:/Windows/Fonts/arialbd.ttf'"
    cap = lambda txt, y: (f"drawtext={FONT}:text='{txt}':fontcolor=white:fontsize=30:box=1:boxcolor=black@0.55:boxborderw=8:"  # noqa: E731
                          f"x=(w-text_w)/2:y={y}")
    if force or not out["x13_dashcam_916_caption"].exists():
        run("-ss", "60", "-i", src("dashcam_urban"), "-t", "12", "-vf",
            f"crop=ih*9/16:ih:iw-ih*9/16:0,scale=406:720,{cap('WATCH THIS DRIVE!', 'h*0.12')},{cap('@viral_clips_daily', 'h*0.86')}",
            *enc(30), str(out["x13_dashcam_916_caption"]))
    if force or not out["x14_retail_tiktok_bars"].exists():
        run("-ss", "40", "-i", src("retail_cctv"), "-t", "12", "-vf",
            f"scale=720:-2,pad=720:1280:0:(1280-ih)/2:black,{cap('CCTV LEAK!!', '120')},{cap('@viral_clips_daily', '1150')}",
            *enc(30), str(out["x14_retail_tiktok_bars"]))
    if force or not out["x15_yonge_916_offgrid"].exists():
        run("-ss", "20", "-i", src("yonge_dundas"), "-t", "12", "-vf",
            "crop=ih*9/16:ih:(iw-ih*9/16)*0.75:0,scale=406:720", *enc(30), str(out["x15_yonge_916_offgrid"]))
    if force or not out["x6_lisbon_holdout"].exists():
        run("-ss", "300", "-i", src("lisbon_parking"), "-t", "15", *enc(23), str(out["x6_lisbon_holdout"]))
    if force or not out["x7_retail_control"].exists():
        run("-ss", "50", "-i", src("retail_cctv"), "-t", "15", *enc(23), str(out["x7_retail_control"]))
    if force or not out["x8_retail_combo"].exists():
        run("-ss", "70", "-i", src("retail_cctv"), "-t", "15", "-vf", "hflip,crop=iw*0.7:ih*0.7,scale=1270:720,setpts=PTS/1.5",
            *enc(30), str(out["x8_retail_combo"]))
    if force or not out["x9_dashcam_offcentre"].exists():
        run("-ss", "30", "-i", src("dashcam_urban"), "-t", "15", "-vf", "crop=iw*0.7:ih*0.7:iw*0.3:ih*0.15,scale=1280:720",
            *enc(28), str(out["x9_dashcam_offcentre"]))
    if force or not out["x10_yonge_lowres_grade"].exists():
        run("-ss", "10", "-i", src("yonge_dundas"), "-t", "15", "-vf", "scale=320:180,eq=brightness=0.12:contrast=1.3:saturation=0",
            *enc(36), str(out["x10_yonge_lowres_grade"]))
    if force or not out["x12_lisbon_titlecard"].exists():
        run("-ss", "66", "-i", src("lisbon_parking"), "-t", "3", *enc(23), str(out["x12_lisbon_titlecard"]))
    return out


if __name__ == "__main__":
    for k, p in make().items():
        print(k, p.stat().st_size // 1024, "KB", round(pk.info(p)["duration"], 2), "s")
