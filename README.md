# webcam-object-detection

Real-time computer vision experiments on a Logitech C270 USB webcam — a hands-on starting point for
broader Physical AI work. CPU-only inference throughout (no NVIDIA GPU involved), using current
(2026) object detection and hand-tracking models from Ultralytics and MediaPipe.

## What's in here

| Script | What it does |
|---|---|
| [`detect.py`](detect.py) | Real-time object detection — Ultralytics YOLO26, COCO's 80 classes |
| [`hand_gesture.py`](hand_gesture.py) | Hand/finger tracking with gesture-triggered system actions (media keys) |
| [`hand_ui.py`](hand_ui.py) | Touchless virtual button panel — hover + pinch-to-click over live video |
| [`list_cameras.py`](list_cameras.py) | Diagnostic: probes camera indices and saves a snapshot from each working one |

## Requirements

- A webcam (developed against a Logitech C270; any OpenCV-compatible UVC webcam should work)
- Python 3.12+
- Windows (the media-key actions in `hand_gesture.py`/`hand_ui.py` use the Windows `ctypes` virtual-key
  API specifically — the detection/tracking scripts themselves are otherwise platform-agnostic)

## Setup

```bash
pip install -r requirements.txt
```

Ultralytics YOLO weights (`*.pt`) auto-download on first run. MediaPipe's task bundles don't —
fetch them manually:

```bash
curl -L -o gesture_recognizer.task "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
curl -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

Then find which OpenCV camera index maps to your physical webcam (a laptop usually has more than
one camera device competing for low indices):

```bash
python list_cameras.py
```

Inspect the saved `snapshot_<index>.jpg` files, and set `CAMERA_INDEX` at the top of whichever
script you're running to match.

## Usage

Run any one script at a time (they each hold the camera exclusively):

```bash
python detect.py          # object detection
python hand_gesture.py    # gesture -> media action
python hand_ui.py         # virtual button panel
```

Press `q` in the video window to quit any of them.

### Gesture → action mapping (`hand_gesture.py`)

| Gesture | Action |
|---|---|
| 👍 Thumb_Up | Volume Up |
| 👎 Thumb_Down | Volume Down |
| ✊ Closed_Fist | Mute/Unmute |
| ✋ Open_Palm | Play/Pause |
| ✌️ Victory | Next Track |
| ☝️ Pointing_Up | Previous Track |

Actions fire once per gesture *change*, not continuously while a gesture is held — flash the
gesture again to repeat the action.

### Virtual button panel (`hand_ui.py`)

Six buttons (Prev / Vol- / Mute / Play / Vol+ / Next) render along the bottom of the video. Point
your index fingertip at a button to hover it, then pinch your thumb and index fingertip together to
click it. Clicks are edge-triggered on the hover+pinch transition, so holding a pinch doesn't
repeat-click.

## Model choices

- **Object detection**: [YOLO26](https://docs.ultralytics.com/models/yolo26/) (Ultralytics, Jan
  2026) — chosen over older YOLOv8/YOLO11 specifically for CPU/edge inference (natively end-to-end,
  no NMS post-processing, ~43% faster CPU inference than YOLO11 at the same tier). `detect.py`
  defaults to the `x` (largest, most accurate) variant; swap `MODEL_NAME` for `yolo26n.pt` or
  `yolo26s.pt` for real-time speed instead of max accuracy.
- **Hand/gesture tracking**: MediaPipe's [Gesture Recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)
  task — ships a pretrained 7-gesture classification head bundled with 21-point hand landmark
  detection, so gesture classification comes for free.
- **Fingertip cursor/click tracking**: MediaPipe's [HandLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
  task instead — `hand_ui.py` only ever needs raw fingertip coordinates and a pinch distance, never
  canonical gesture classes, so it uses the narrower/cheaper tool rather than paying for gesture
  classification it doesn't use.

## Architecture notes

`camera_utils.py` and `media_actions.py` hold shared helpers (camera/window setup, Windows
media-key simulation via `ctypes`) used across the three interactive scripts. See
[`CLAUDE.md`](CLAUDE.md) for full technical detail, including hardware-specific gotchas (camera
index mapping, a global-Python-env dependency conflict hit while adding MediaPipe, and a stale
background-process gotcha) discovered while building this.

## Roadmap

`hand_gesture.py`'s `trigger_action_for_gesture()` and `hand_ui.py`'s button-click handling are
deliberately the single seam where a gesture/click currently triggers a media-key press — the
natural next step is swapping that for a real robot command (serial/MQTT) once there's actual
hardware to drive, as part of a broader Physical AI learning path.
