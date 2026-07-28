import cv2
import numpy as np
import pyautogui
import logging
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32

def find_window_by_title(keywords):
    hwnd = None
    def enum_callback(handle, _):
        nonlocal hwnd
        length = user32.GetWindowTextLengthW(handle) + 1
        buffer = ctypes.create_unicode_buffer(length)
        user32.GetWindowTextW(handle, buffer, length)
        title = buffer.value
        for kw in keywords:
            if kw in title:
                hwnd = handle
                return False
        return True
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    callback = enum_proc(enum_callback)
    user32.EnumWindows(callback, 0)
    return hwnd

class Capture:
    def __init__(self):
        self.window_offset_x = 0
        self.window_offset_y = 0

    def get_window_client_rect(self):
        hwnd = find_window_by_title(["扫雷", "Minesweeper", "minesweeper", "扫雷游戏"])
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
