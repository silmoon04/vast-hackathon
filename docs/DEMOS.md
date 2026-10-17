# Demo gallery

Animated previews from existing demo videos. These loops are trimmed, resized and converted to GIF at
8 frames per second; the original annotations remain visible. They show saved outputs, with the evaluation
limits described below and in each project's README.

## Rewind a street scene in seconds

![Street Time Machine synopsis with timestamped moving-object ghosts crossing a street](demos/street-time-machine.gif)

Street Time Machine compresses a recording into a synopsis by placing tracked moving objects onto a shared
background. This is an eight-second extract from the saved Yonge-Dundas synopsis. It is a time composite:
the ghosts come from different source times and did not all occur together. Heads are blurred, and the
synopsis omits or caps some tracks.

[Code and usage](../street-time-machine/README.md). Footage: Raysonho / Open Grid Scheduler / Grid Engine, CC0.

## Draw a path to search video

![MIMIC top motion match following a worker around a U-turn with a tracked path overlay](demos/mimic-path-search.gif)

MIMIC matches a drawn motion path against tracked objects. This is the exported top match for a U-turn
query over prepared tracks, with the box, trajectory and source timestamp retained. The GIF shows a search
result rather than the drawing interface. Lower-ranked results can be false matches; costs are similarity
scores, not identification probabilities.

[Code and usage](../prototypes/07_mimic/README.md). Footage: Intel Corporation sample videos, CC BY 4.0.

## Detect privacy-mask drift

![Private View privacy masks becoming misaligned as the camera moves, with changing exposure indicators](demos/private-view.gif)

Private View compares static mask polygons with estimated window positions after a camera moves. This
excerpt uses real hand-held drift without injected motion and includes the transition to the BREACH state.
The windows and masks are operator-drawn test zones; exposure is estimated, and this run has no ground-truth
labels.

[Code and usage](../prototypes/15_private_view/README.md). Footage: OathOn, CC BY-SA 4.0.

## Release only the requested subject

![Release Desk exported video following the selected shopper while masking the surrounding store](demos/release-desk.gif)

Release Desk keeps the selected subject visible while masking the rest of the frame. This loop comes
directly from the exported video for a demonstration request. Separate detectors check exported frames;
zero detected leaks is detector evidence, and frames requiring human review remain listed in the report.

[Code and usage](../prototypes/01_release_desk/README.md). Footage: HDCCTV Cameras, catalogued as public domain.

## Inspect stopped traffic in context

![Lane Call stopped-vehicle alert with a tracked car and source timestamp in moving traffic footage](demos/lane-call.gif)

Lane Call ranks stopped-vehicle alerts and provides short evidence clips. This is a saved alert from real
signal-controlled traffic. Lane polygons are configured by hand; an IN LANE label identifies position,
and stopped traffic can be a queue. The displayed confidence is a heuristic score.

[Code and usage](../prototypes/14_lane_call/README.md). Footage: Raysonho / Open Grid Scheduler / Grid Engine, CC0.

## Footage credits

The demos extract and arrange footage frames with masks, boxes, tracks, timestamps and other overlays.
The GIFs preserve those annotations while reducing resolution and frame rate. The Private View GIF,
including its overlays, is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
These image terms do not license the repository's source code.

- **Intel Corporation:** the U-turn result uses `worker-zone-detection.mp4` from
  [Intel IoT sample videos](https://github.com/intel-iot-devkit/sample-videos),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **HDCCTV Cameras:** [Retail store CCTV footage](https://commons.wikimedia.org/wiki/File:HD_CCTV_Camera_video_3MP_4MP_iProx_CCTV_HDCCTVCameras.net_retail_store.webm),
  catalogued as public domain. Used in Release Desk.
- **OathOn:** [Corporation Street, Rochester, at night](https://commons.wikimedia.org/wiki/File:Corporation_Street,_Rochester,_Kent,_England,_at_night.webm),
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Used in Private View.
- **Raysonho / Open Grid Scheduler / Grid Engine:** [Diagonal crossing at Yonge-Dundas](https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm),
  [CC0](https://creativecommons.org/publicdomain/zero/1.0/). Used in Street Time Machine and Lane Call.

Source records are retained in the [video catalogue](../prototypes/videos.json),
[Intel clip definitions](../prototypes/protokit.py) and [camera examples](../street-time-machine/cams.example.json).
