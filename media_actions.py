"""
media_actions.py

Windows system-wide media key simulation via ctypes (stdlib only, no extra
dependency). Shared by hand_gesture.py and hand_ui.py.
"""

import ctypes

# Windows virtual-key codes for multimedia keys (winuser.h).
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3


def send_media_key(vk_code: int) -> None:
    """Simulate a system-wide press-and-release of a Windows virtual media key."""
    user32 = ctypes.windll.user32
    user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
    user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
