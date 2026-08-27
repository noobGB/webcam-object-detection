"""
camera_utils.py

Shared webcam and display-window helpers used by detect.py, hand_gesture.py,
and hand_ui.py - factored out once a third script needed the identical setup.
"""

import cv2


def open_camera(index: int) -> cv2.VideoCapture:
    """Open the webcam at the given OpenCV device index using DirectShow."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {index}")
    return cap


def create_resizable_window(title: str, width: int = 1920, height: int = 1080) -> None:
    """Create a normal, drag-resizable display window, sized to Full HD by default.

    The captured frame itself stays at its native (small) resolution -
    this only stretches how it's displayed, not how much detail is captured.
    """
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, width, height)
