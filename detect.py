"""
detect.py

Real-time object detection on a live webcam feed using Ultralytics YOLO26x
(the extra-large, most accurate variant — 57.5 mAP vs. small's 48.6, but
~526ms/frame on CPU per Ultralytics' published benchmark, so expect ~1-2 fps
on this machine, not smooth real-time. Useful for seeing near-max detection
quality on a relatively static scene; drop back to yolo26s.pt/yolo26n.pt for
live tracking).

Opens the Logitech C270 (OpenCV device index 1, confirmed via
list_cameras.py), runs each frame through the model, and displays the
annotated video feed in a window. Press 'q' to quit.

Usage:
    python detect.py
"""

import cv2
from ultralytics import YOLO

from camera_utils import create_resizable_window, open_camera

CAMERA_INDEX = 1
MODEL_NAME = "yolo26x.pt"
CONFIDENCE_THRESHOLD = 0.5
WINDOW_TITLE = "YOLO26x - Live Detection (press 'q' to quit)"


def run_detection_loop(model: YOLO, cap: cv2.VideoCapture) -> None:
    """Read frames from the camera, run detection, and display results until 'q' is pressed."""
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera; stopping.")
            break

        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow(WINDOW_TITLE, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    model = YOLO(MODEL_NAME)
    cap = open_camera(CAMERA_INDEX)
    create_resizable_window(WINDOW_TITLE)
    try:
        run_detection_loop(model, cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
