"""
list_cameras.py

Probes camera indices 0-4 via OpenCV (DirectShow backend) and saves one
snapshot per working index, so we can visually identify which index maps
to the physical Logitech C270 versus the laptop's built-in/IR cameras.

Usage:
    python list_cameras.py
Then inspect the saved snapshot_<index>.jpg files in this folder.
"""

import cv2

MAX_INDEX_TO_PROBE = 5
SNAPSHOT_PREFIX = "snapshot_"


def probe_camera(index: int) -> bool:
    """Try to open a camera at the given index and save one frame to disk.

    Returns True if a frame was successfully captured, False otherwise.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return False

    ok, frame = cap.read()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    if not ok or frame is None:
        return False

    filename = f"{SNAPSHOT_PREFIX}{index}.jpg"
    cv2.imwrite(filename, frame)
    print(f"index {index}: OK, {int(width)}x{int(height)} -> saved {filename}")
    return True


def main() -> None:
    found_any = False
    for index in range(MAX_INDEX_TO_PROBE):
        if probe_camera(index):
            found_any = True
        else:
            print(f"index {index}: not available")

    if not found_any:
        print("No working camera indices found.")


if __name__ == "__main__":
    main()
