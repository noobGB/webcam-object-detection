"""
hand_gesture.py

Real-time hand/finger tracking and gesture-triggered system actions on a
live webcam feed, using MediaPipe's Gesture Recognizer task (hand landmark
detection + a gesture classification head in one model bundle).

Tracks up to two hands, draws the 21 landmarks per hand, overlays the
recognized gesture near the wrist, and fires a real system action (media
key press, e.g. volume/play-pause/track-skip) the moment a hand's
recognized gesture changes to one of the mapped gestures in GESTURE_ACTIONS.
Actions are sent system-wide via the Windows virtual-key API (ctypes,
stdlib only - no extra dependency), so they land wherever Windows' media
keys normally would (Spotify, browser video, system volume, etc).

Opens the Logitech C270 (OpenCV device index 1, confirmed via
list_cameras.py). Press 'q' to quit.

Usage:
    python hand_gesture.py
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from camera_utils import create_resizable_window, open_camera
from media_actions import (
    VK_MEDIA_NEXT_TRACK,
    VK_MEDIA_PLAY_PAUSE,
    VK_MEDIA_PREV_TRACK,
    VK_VOLUME_DOWN,
    VK_VOLUME_MUTE,
    VK_VOLUME_UP,
    send_media_key,
)

CAMERA_INDEX = 1
MODEL_PATH = "gesture_recognizer.task"
MIN_HAND_DETECTION_CONFIDENCE = 0.5
NUM_HANDS = 2
WINDOW_TITLE = "MediaPipe Gesture Recognizer (press 'q' to quit)"

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# Maps a recognized gesture name to (action label shown on screen, virtual-key code).
# Gestures not listed here (e.g. ILoveYou) are tracked and displayed but trigger no action.
GESTURE_ACTIONS: dict[str, tuple[str, int]] = {
    "Thumb_Up": ("Volume Up", VK_VOLUME_UP),
    "Thumb_Down": ("Volume Down", VK_VOLUME_DOWN),
    "Closed_Fist": ("Mute/Unmute", VK_VOLUME_MUTE),
    "Open_Palm": ("Play/Pause", VK_MEDIA_PLAY_PAUSE),
    "Victory": ("Next Track", VK_MEDIA_NEXT_TRACK),
    "Pointing_Up": ("Previous Track", VK_MEDIA_PREV_TRACK),
}


def create_recognizer() -> vision.GestureRecognizer:
    """Load the gesture recognizer model in synchronous VIDEO mode."""
    options = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
    )
    return vision.GestureRecognizer.create_from_options(options)


def draw_landmarks(frame, hand_landmarks) -> None:
    """Draw one hand's 21 landmarks and connections onto the frame in place."""
    vision.drawing_utils.draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS)


def draw_gesture_label(frame, hand_landmarks, gesture_name: str, confidence: float, action_label: str | None) -> None:
    """Overlay the recognized gesture name (and mapped action, if any) near the wrist landmark."""
    height, width = frame.shape[:2]
    wrist = hand_landmarks[0]
    x, y = int(wrist.x * width), int(wrist.y * height)
    label = f"{gesture_name} ({confidence:.2f})"
    if action_label:
        label += f" -> {action_label}"
    cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def trigger_action_for_gesture(hand_index: int, gesture_name: str) -> str | None:
    """Fire the system action mapped to this gesture, if any; return its display label."""
    action = GESTURE_ACTIONS.get(gesture_name)
    if action is None:
        return None
    action_label, vk_code = action
    send_media_key(vk_code)
    print(f"[action] hand {hand_index}: {gesture_name} -> {action_label}")
    return action_label


def run_tracking_loop(recognizer: vision.GestureRecognizer, cap: cv2.VideoCapture) -> None:
    """Read frames, run gesture recognition, display results, and trigger actions on gesture changes."""
    start_time = time.time()
    last_gesture_per_hand: dict[int, str] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera; stopping.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
            draw_landmarks(frame, hand_landmarks)

            if hand_index < len(result.gestures) and result.gestures[hand_index]:
                top_gesture = result.gestures[hand_index][0]
                gesture_name = top_gesture.category_name

                action_label = None
                if last_gesture_per_hand.get(hand_index) != gesture_name:
                    action_label = trigger_action_for_gesture(hand_index, gesture_name)
                    last_gesture_per_hand[hand_index] = gesture_name

                draw_gesture_label(frame, hand_landmarks, gesture_name, top_gesture.score, action_label)

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    recognizer = create_recognizer()
    cap = open_camera(CAMERA_INDEX)
    create_resizable_window(WINDOW_TITLE)
    try:
        run_tracking_loop(recognizer, cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()


if __name__ == "__main__":
    main()
