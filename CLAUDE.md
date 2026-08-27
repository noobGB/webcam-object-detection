# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Learning experiments in real-time computer vision using a Logitech C270 USB webcam, as a starting
point for broader Physical AI work: object detection (Ultralytics YOLO26), hand/finger tracking +
gesture-triggered actions (MediaPipe Gesture Recognizer), and a touchless virtual button panel
(MediaPipe HandLandmarker + hover/pinch-to-click). CPU-only inference (this machine has no NVIDIA
GPU — Intel Iris Xe integrated graphics only), so model choice and export format matter more here
than they would on a GPU box.

## Commands

- Run the live object detection demo: `python detect.py` (press `q` in the window to quit)
- Run the live hand-tracking/gesture demo: `python hand_gesture.py` (press `q` to quit)
- Run the live hand-tracking virtual button panel: `python hand_ui.py` (hover fingertip over a
  button, pinch thumb+index to click; press `q` to quit)
- Re-identify which OpenCV camera index maps to the physical C270 (vs. the laptop's built-in/IR
  cameras) if indices ever shift: `python list_cameras.py`, then inspect the saved
  `snapshot_<index>.jpg` files
- Install/update deps: `pip install -r requirements.txt`
- **Only run one camera script at a time** — the C270 is a single physical device; a second script
  opening it while another holds it open fails with `RuntimeError: Could not open camera at index 1`.
  If that happens with nothing obviously running, check for an orphaned process first (see below)
  before assuming it's a code bug.

## Hardware / environment facts

- OpenCV device index **1** (via `cv2.CAP_DSHOW`) is the physical Logi C270. Index 0 and other
  indices did not open on this machine — the laptop also exposes an "Integrated Camera" and
  "Integrated IR Camera" as separate PnP devices, so don't assume index 0 is the external webcam.
  Re-run `list_cameras.py` if a camera stops being detected or a new one is added.
- No CUDA — `torch`/`ultralytics` run in CPU mode. Ultralytics model size is a direct speed/accuracy
  trade (n/s/m/l/x); step up in size only if accuracy is the bottleneck, not speed.
- **This machine's Python is one shared global install, not a per-project venv** — installing
  `mediapipe` here bumped `numpy` 1.26.4→2.5.2 (a hard dependency), which broke the already-installed
  `torch` at runtime (`torch.from_numpy` failing with "Numpy is not available", since the old torch
  build predated numpy 2.x's ABI). Fixed by upgrading `torch`→2.13.0, not by pinning numpy back down
  (mediapipe needs numpy 2.x). See `technical_notes.md` (workspace root) for the full incident,
  including a second gotcha: `opencv-python` and `opencv-contrib-python` (which mediapipe requires)
  share the same `cv2` install path and conflict if both are installed — keep only
  `opencv-contrib-python`, never both.
- **Stopping a backgrounded camera script doesn't always kill the actual `python.exe`** — hit this
  with the harness's own scoped-stop mechanism leaving an orphaned `python hand_gesture.py` process
  running (and holding the camera) after being told to stop. If a camera script fails to open the
  device with nothing obviously running, check for a stray `python.exe` first (e.g.
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select ProcessId,CommandLine`) before
  assuming it's a code bug.

## Architecture

- `camera_utils.py` — shared `open_camera()` (C270 via `cv2.CAP_DSHOW`) and
  `create_resizable_window()` (Full HD-sized, drag-resizable `cv2.WINDOW_NORMAL` — deliberately not
  `WINDOW_FULLSCREEN`, which locks borderless with no drag-resize). Factored out once three scripts
  needed the identical setup.
- `media_actions.py` — shared `send_media_key(vk_code)` + Windows virtual-key constants
  (`VK_VOLUME_UP` etc.), simulating a system-wide media key press via `ctypes` (stdlib only, no
  extra dependency) — lands wherever Windows' media keys normally would (Spotify, browser video,
  system volume).
- `detect.py` — real-time object detection: runs each frame through an Ultralytics YOLO model,
  overlays detections with `results[0].plot()`. Config constants (model weights file, confidence
  threshold) are at the top of the file.
- `hand_gesture.py` — real-time hand/finger tracking with gesture-triggered actions: runs each
  frame through MediaPipe's `GestureRecognizer` task (21 hand landmarks + a 7-class gesture head —
  Thumb_Up, Thumb_Down, Victory, Pointing_Up, Closed_Fist, Open_Palm, ILoveYou), draws the landmark
  skeleton, and the moment a hand's recognized gesture *changes* to one of the gestures in
  `GESTURE_ACTIONS`, fires the mapped media action. Mapping: Thumb_Up→Volume Up, Thumb_Down→Volume
  Down, Closed_Fist→Mute/Unmute, Open_Palm→Play/Pause, Victory→Next Track, Pointing_Up→Previous
  Track. Actions fire once per gesture change, not continuously while held (holding Thumb_Up
  doesn't spam volume-up — flash the gesture again to repeat it). `trigger_action_for_gesture` is
  the single seam that owns "gesture happened, do a thing" — the natural spot to swap in a real
  robot command (serial/MQTT) once there's hardware to drive, per the robotics/Physical AI
  direction.
- `hand_ui.py` — real-time hand-tracking **virtual button panel**: runs each frame through
  MediaPipe's `HandLandmarker` task (landmarks only, no gesture classification needed for this one)
  and draws 6 clickable buttons (Prev/Vol-/Mute/Play/Vol+/Next) along the bottom of the frame.
  Tracks the index fingertip (landmark 8) as a cursor for hover, and the thumb tip (landmark 4) to
  detect a pinch (Euclidean pixel distance < `PINCH_THRESHOLD_PX`) as the "click" gesture — a click
  fires once on the hover+pinch *transition* (edge-triggered via `was_pinching`, same one-shot
  pattern as `hand_gesture.py`'s gesture-change detection), not continuously while pinched. Buttons
  render with a semi-transparent fill (`cv2.addWeighted` overlay blend) that changes color for
  default/hover/click-flash states.
- `list_cameras.py` — standalone diagnostic that probes camera indices 0-4 and saves one snapshot
  per working index, used to (re-)determine which index is the C270.
- Model weights (`*.pt`, `*.onnx`, `*.task`) are downloaded/fetched on first run and gitignored —
  not checked into the repo. MediaPipe's `.task` bundles aren't auto-downloaded like Ultralytics
  weights are — fetch manually from Google's model bundle storage, e.g.:
  `curl -L -o gesture_recognizer.task "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"`
  (swap `gesture_recognizer` for `hand_landmarker` for the landmarks-only model `hand_ui.py` uses).

## Model choice

- **Object detection**: currently set to **YOLO26x** in `detect.py` (Ultralytics' largest/most
  accurate variant, 57.5 mAP) for exploring max detection quality — expect ~1-2 fps on this CPU, not
  real-time. **YOLO26s** (48.6 mAP, ~10 fps) is the better default for live/interactive use; **YOLO26n**
  (40.9 mAP, ~25 fps) is fastest but only reliably detects large/common classes (person, chair).
  Swap `MODEL_NAME` in `detect.py` to change tiers. YOLO26 (Jan 2026) was chosen over older
  YOLOv8/YOLO11 nano models specifically for CPU/edge inference: natively end-to-end (no NMS
  post-processing step), DFL removed, up to 43% faster CPU inference than YOLO11n at the same tier.
- **Hand/gesture tracking** (`hand_gesture.py`): MediaPipe's Gesture Recognizer task, chosen over
  rolling a custom finger-counting heuristic on top of raw landmarks — it ships a pretrained
  7-gesture classification head bundled with the landmark model, so gesture classification comes
  for free rather than needing hand-written geometry rules.
- **Fingertip cursor/click tracking** (`hand_ui.py`): MediaPipe's HandLandmarker task instead of
  GestureRecognizer, even though GestureRecognizer's result also includes landmarks — deliberately
  picked the narrower tool since this script only ever needs raw fingertip coordinates (for hover)
  and a hand-computed pinch distance (for click), never the canonical gesture classes, so running
  the extra gesture-classification head would just be wasted CPU work.
- **MediaPipe API note**: the installed `mediapipe` (1.0.1) has been restructured — the old
  `mediapipe.solutions.*` and `mediapipe.framework.formats.landmark_pb2` APIs referenced in older
  tutorials/StackOverflow answers **do not exist** in this install (only a `mediapipe.tasks`
  submodule ships). Use `mediapipe.tasks.python.vision.drawing_utils.draw_landmarks(...)` (takes the
  landmark list directly, no protobuf wrapping needed) and
  `mediapipe.tasks.python.vision.HandLandmarksConnections.HAND_CONNECTIONS` instead.
