"""
hand_ui.py

Real-time hand-tracking video with an overlaid row of clickable media
buttons (Previous, Volume Down, Mute/Unmute, Play/Pause, Volume Up, Next).
Point your index fingertip at a button to hover it, then pinch your thumb
and index fingertip together to "click" it - a touchless virtual button
panel, using MediaPipe's HandLandmarker for the raw 21-point hand skeleton
(no gesture classification needed here, just fingertip position).

Click detection is edge-triggered on the hover+pinch transition, so holding
a pinch doesn't repeat-fire the action - release and pinch again to repeat.

Opens the Logitech C270 (OpenCV device index 1, confirmed via
list_cameras.py). Press 'q' to quit.

Usage:
    python hand_ui.py
"""

import math
import time
from dataclasses import dataclass

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
MODEL_PATH = "hand_landmarker.task"
MIN_HAND_DETECTION_CONFIDENCE = 0.5
NUM_HANDS = 1
WINDOW_TITLE = "Hand UI - hover + pinch to click (press 'q' to quit)"

INDEX_FINGERTIP = 8
THUMB_TIP = 4
PINCH_THRESHOLD_PX = 40
CLICK_FLASH_SECONDS = 0.3

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# (button label, action label shown on hover, virtual-key code) - left to right.
BUTTON_DEFS = [
    ("PREV", "Previous Track", VK_MEDIA_PREV_TRACK),
    ("VOL-", "Volume Down", VK_VOLUME_DOWN),
    ("MUTE", "Mute/Unmute", VK_VOLUME_MUTE),
    ("PLAY", "Play/Pause", VK_MEDIA_PLAY_PAUSE),
    ("VOL+", "Volume Up", VK_VOLUME_UP),
    ("NEXT", "Next Track", VK_MEDIA_NEXT_TRACK),
]

BUTTON_HEIGHT = 70
BUTTON_GAP = 10
BUTTON_BOTTOM_MARGIN = 20


@dataclass
class Button:
    x: int
    y: int
    width: int
    height: int
    label: str
    action_label: str
    vk_code: int

    def contains(self, px: int, py: int) -> bool:
        """Return whether the given pixel point falls inside this button."""
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height


def build_buttons(frame_width: int, frame_height: int) -> list[Button]:
    """Lay out the button row evenly spaced along the bottom of the frame."""
    count = len(BUTTON_DEFS)
    width = (frame_width - BUTTON_GAP * (count + 1)) // count
    y = frame_height - BUTTON_HEIGHT - BUTTON_BOTTOM_MARGIN
    buttons = []
    for i, (label, action_label, vk_code) in enumerate(BUTTON_DEFS):
        x = BUTTON_GAP + i * (width + BUTTON_GAP)
        buttons.append(Button(x, y, width, BUTTON_HEIGHT, label, action_label, vk_code))
    return buttons


def create_landmarker() -> vision.HandLandmarker:
    """Load the hand landmarker model in synchronous VIDEO mode."""
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_button(frame, button: Button, is_hovered: bool, is_clicked: bool) -> None:
    """Draw one button with a fill/border style reflecting its hover/click state."""
    top_left = (button.x, button.y)
    bottom_right = (button.x + button.width, button.y + button.height)

    if is_clicked:
        fill_color, border_color = (0, 160, 0), (0, 255, 0)
    elif is_hovered:
        fill_color, border_color = (0, 130, 180), (0, 255, 255)
    else:
        fill_color, border_color = (60, 60, 60), (150, 150, 150)

    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, fill_color, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.rectangle(frame, top_left, bottom_right, border_color, 2)

    text_size = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = button.x + (button.width - text_size[0]) // 2
    text_y = button.y + (button.height + text_size[1]) // 2
    cv2.putText(frame, button.label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def get_pixel_point(landmark, frame_width: int, frame_height: int) -> tuple[int, int]:
    """Convert a normalized (0-1) landmark coordinate to pixel coordinates."""
    return int(landmark.x * frame_width), int(landmark.y * frame_height)


def run_ui_loop(landmarker: vision.HandLandmarker, cap: cv2.VideoCapture) -> None:
    """Read frames, track the hand, and drive the button panel from fingertip hover/pinch."""
    start_time = time.time()
    buttons: list[Button] | None = None
    was_pinching = False
    last_clicked_button: Button | None = None
    last_click_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera; stopping.")
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        if buttons is None:
            buttons = build_buttons(frame_width, frame_height)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        hovered_button = None
        is_pinching = False

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            vision.drawing_utils.draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS)

            cursor_point = get_pixel_point(hand_landmarks[INDEX_FINGERTIP], frame_width, frame_height)
            thumb_point = get_pixel_point(hand_landmarks[THUMB_TIP], frame_width, frame_height)
            pinch_distance = math.hypot(cursor_point[0] - thumb_point[0], cursor_point[1] - thumb_point[1])
            is_pinching = pinch_distance < PINCH_THRESHOLD_PX

            for button in buttons:
                if button.contains(*cursor_point):
                    hovered_button = button
                    break

            if hovered_button and is_pinching and not was_pinching:
                send_media_key(hovered_button.vk_code)
                print(f"[click] {hovered_button.action_label}")
                last_clicked_button = hovered_button
                last_click_time = time.time()

            cursor_color = (0, 255, 0) if is_pinching else (0, 200, 255)
            cv2.circle(frame, cursor_point, 12, cursor_color, -1)

        was_pinching = is_pinching

        flash_active = (time.time() - last_click_time) < CLICK_FLASH_SECONDS
        for button in buttons:
            is_clicked = flash_active and button is last_clicked_button
            draw_button(frame, button, is_hovered=(button is hovered_button), is_clicked=is_clicked)

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    landmarker = create_landmarker()
    cap = open_camera(CAMERA_INDEX)
    create_resizable_window(WINDOW_TITLE)
    try:
        run_ui_loop(landmarker, cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
