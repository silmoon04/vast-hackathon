"""Precompute (cache) YOLO26s+ByteTrack tracks for every test clip, and poses for the fall clips."""
import time
import protokit as pk
SKIP = {"bolt_detection", "driver_action_recognition"}
FALLS = ["skate_fall", "floor_lie", "assist_elderly", "help_sit", "sit_down", "classroom", "one_by_one_person_detection"]
t0 = time.time()
for n in sorted(pk.videos(), key=lambda n: pk.info(n)["duration"]):
    if n in SKIP:
        continue
    pk.tracks(n)
for n in FALLS:
    print("pose", n, len(pk.poses(n)), flush=True)
print("all done in", round(time.time() - t0), "s")
