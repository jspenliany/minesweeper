import cv2
import numpy as np
import pyautogui
import logging
import ctypes
from ctypes import wintypes
from src.timer import timer

user32 = ctypes.windll.user32

# Common Minesweeper window title keywords
# Note: English keywords are case-sensitive to avoid matching folder names like "minesweeper-bot"
MINESWEEPER_KEYWORDS = [
    "扫雷",                    # Chinese
    "扫雷游戏",                # Chinese variant
    "Minesweeper",             # English (capitalized — won't match "minesweeper-bot")
    "Microsoft Minesweeper",   # MS Store version
]

# Window classes that are definitely NOT Minesweeper
EXCLUDED_CLASSES = {"CabinetWClass", "ExploreWClass", "Progman", "WorkerW", "ConsoleWindowClass"}

@timer
def find_window_by_title(keywords):
    candidates = []
    def enum_callback(handle, _):
        length = user32.GetWindowTextLengthW(handle) + 1
        buffer = ctypes.create_unicode_buffer(length)
        user32.GetWindowTextW(handle, buffer, length)
        title = buffer.value
        for kw in keywords:
            if kw in title:
                # Skip non-game windows by class name
                cls_buf = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(handle, cls_buf, 256)
                if cls_buf.value in EXCLUDED_CLASSES:
                    return True
                if user32.IsWindowVisible(handle):
                    rect = wintypes.RECT()
                    user32.GetClientRect(handle, ctypes.byref(rect))
                    if rect.right > 200 and rect.bottom > 200:
                        candidates.append((handle, title))
                break
        return True
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    callback = enum_proc(enum_callback)
    user32.EnumWindows(callback, 0)
    # Sort: prefer windows whose title starts with a keyword (main game window)
    # over other matches (e.g. a dialog that happens to contain the word)
    def score(h):
        t = h[1]
        for kw in keywords:
            if t.startswith(kw) or t == kw:
                return 2
            if kw in t:
                return 1
        return 0
    candidates.sort(key=lambda h: -score(h))
    return candidates[0][0] if candidates else None

class Capture:
    def __init__(self):
        self.window_offset_x = 0
        self.window_offset_y = 0

    def get_window_client_rect(self):
        hwnd = find_window_by_title(MINESWEEPER_KEYWORDS)
        if not hwnd:
            return None
        rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(pt))
        return {
            'left': pt.x,
            'top': pt.y,
            'width': rect.right,
            'height': rect.bottom
        }

    @timer
    def get_screenshot(self):
        win_rect = self.get_window_client_rect()
        if win_rect and win_rect['width'] > 200 and win_rect['height'] > 200:
            self.window_offset_x = win_rect['left']
            self.window_offset_y = win_rect['top']
            screenshot = pyautogui.screenshot(region=(
                win_rect['left'], win_rect['top'],
                win_rect['width'], win_rect['height']
            ))
            logging.info(f"Captured window client area: ({win_rect['left']}, {win_rect['top']}) {win_rect['width']}x{win_rect['height']}")
        else:
            self.window_offset_x = 0
            self.window_offset_y = 0
            screenshot = pyautogui.screenshot()
            logging.info("Window not found, captured full screen.")
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def to_screen(self, x, y):
        return x + self.window_offset_x, y + self.window_offset_y
