# Posture Guard - Proof of Concept

Real-time "are you sitting too close to the screen" detector, built as
Phase 1 of a larger posture-correction app for the Congressional App
Challenge. This is intentionally minimal: it only checks screen distance
using face-width detection. Full slouch/posture detection (via body
pose keypoints) is a planned Phase 2, not part of this POC.

## How it works

1. Your webcam feed is scanned every frame for a face using OpenCV's
   Haar Cascade face detector (bundled in `src/main/resources`).
2. For the first ~30 frames a face is detected, the app averages your
   face's width in pixels to set a personal "comfortable distance"
   baseline.
3. After that, if your face width grows more than 20% past your
   baseline for a sustained stretch (meaning you leaned in toward the
   screen), it prints an alert to the console and draws a red box
   around your face in the video window. Green box = good distance,
   yellow box = still calibrating.

## Requirements

- JDK 11 or newer
- Maven 3.6+
- A webcam
- macOS/Windows/Linux (javacv-platform bundles native OpenCV binaries
  for all three, so there's no separate OpenCV install needed)

## Running it

```bash
mvn compile exec:java
```

The first run will take a while since Maven needs to download the
javacv-platform dependency (~150-200MB, it bundles native libraries for
every OS so your build stays portable). After that, builds are fast.

Sit at your normal, comfortable distance when the window opens and
hold still for a couple seconds while it calibrates. Then lean in
toward the screen to test that the alert fires.

## Project layout

```
posture-guard-poc/
├── pom.xml
├── src/main/java/com/posturewatch/ScreenDistanceMonitor.java   <- all the logic
└── src/main/resources/haarcascade_frontalface_default.xml       <- pretrained face detector
```

## Known limitations (this is a POC, not a finished app)

- Only tracks screen distance, not posture/slouching yet.
- Baseline calibration is naive (just an average of the first 30
  frames) -- if you're mid-lean during calibration, your baseline will
  be off. A "press SPACE to calibrate" step would fix this in a later
  version.
- Face detection can be flaky in low light or at extreme angles --
  test in the actual room/lighting you plan to demo in.
- Alerts currently just print to the console; a real notification/UI
  layer is a next step.

## Suggested next steps (Phase 2+)

1. Swap the naive auto-calibration for a manual "press SPACE when
   you're comfortable" trigger.
2. Add body pose keypoints (e.g. a MoveNet model run through OpenCV's
   DNN module) to detect forward-head/slouching posture, not just
   distance.
3. Build a small JavaFX dashboard showing a live status indicator plus
   a per-session summary (% time in good posture/distance).
4. Log session data (timestamps + alert counts) to a local file or
   SQLite DB for the summary view.
