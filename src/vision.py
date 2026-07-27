import cv2
import numpy as np
import pyautogui
import logging
import os
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class Vision:
    def __init__(self, assets_path="D:/workspace/minesweeper-bot/assets"):
        self.assets_path = assets_path
        self.templates = {}
        self.window_offset_x = 0
        self.window_offset_y = 0
        self.load_templates()

    def load_templates(self):
        anchor_dir = os.path.join(self.assets_path, "anchors")
        if os.path.exists(anchor_dir):
            for file in os.listdir(anchor_dir):
                path = os.path.join(anchor_dir, file)
                self.templates[file] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        tile_dir = os.path.join(self.assets_path, "tiles")
        if os.path.exists(tile_dir):
            for file in os.listdir(tile_dir):
                path = os.path.join(tile_dir, file)
                self.templates[file] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        logging.info(f"Loaded {len(self.templates)} templates.")

    def get_window_client_rect(self):
        """Get the client area rectangle of the Minesweeper window on screen"""
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
        """Capture the Minesweeper window client area (or full screen if window not found)"""
        win_rect = self.get_window_client_rect()
        if win_rect and win_rect['width'] > 0 and win_rect['height'] > 0:
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

    def _screen_to_client(self, x, y):
        """Convert relative window coordinates to absolute screen coordinates"""
        return x + self.window_offset_x, y + self.window_offset_y

    def find_image(self, target_img, template_name, threshold=0.8):
        template = self.templates.get(template_name)
        if template is None:
            return None

        res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val >= threshold:
            h, w = template.shape[:2]
            return max_loc[0], max_loc[1], w, h
        return None

    def calibrate_grid(self):
        """Find the game board by scanning the entire screenshot for closed tiles.
        Finds the top-leftmost closed tile to establish the grid origin.
        """
        screen = self.get_screenshot()
        debug_img = screen.copy()
        
        tile_template = self.templates.get("closed_tile.png")
        if tile_template is None:
            logging.error("closed_tile.png missing in assets.")
            return None
        tw, th = tile_template.shape[1], tile_template.shape[0]
        logging.info(f"Cell template size: {tw}x{th}")
        
        # Scan entire screenshot for closed tiles
        res = cv2.matchTemplate(screen, tile_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(res >= 0.8)
        
        if len(locations[0]) == 0:
            logging.error("Could not find any closed tiles in the screenshot.")
            cv2.imwrite("debug_no_tile.png", debug_img)
            return None
        
        # Collect all match positions (top-left corners) and find the top-leftmost tile
        matches = list(zip(locations[1], locations[0]))  # (x, y) pairs
        matches.sort(key=lambda p: (p[1], p[0]))  # Sort by y first (topmost), then x (leftmost)
        tile_x, tile_y = matches[0]
        
        # Draw anchor if found (for reference only)
        anchor = self.find_image(screen, "board_tl.png")
        if anchor:
            ax, ay, aw, ah = anchor
            cv2.rectangle(debug_img, (ax, ay), (ax + aw, ay + ah), (0, 255, 0), 2)
            cv2.putText(debug_img, "Anchor", (ax, ay - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            logging.info(f"Anchor found at ({ax}, {ay}) for reference")
        
        # Validate: check if the entire board fits within the screenshot
        board_right = tile_x + (30 * tw)
        board_bottom = tile_y + (16 * th)
        if board_right > screen.shape[1] or board_bottom > screen.shape[0]:
            logging.warning(f"Board extends beyond screenshot: board right={board_right}, bottom={board_bottom}, "
                           f"screenshot size=({screen.shape[1]}x{screen.shape[0]})")
            cv2.putText(debug_img, f"BOARD RIGHT={board_right}", (tile_x, tile_y + th + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        origin_x, origin_y = self._screen_to_client(tile_x + tw // 2, tile_y + th // 2)
        
        cv2.rectangle(debug_img, (tile_x, tile_y), (tile_x + tw, tile_y + th), (0, 255, 255), 2)
        cv2.circle(debug_img, (tile_x + tw // 2, tile_y + th // 2), 4, (0, 0, 255), -1)
        cv2.putText(debug_img, "Cell(0,0)", (tile_x + tw + 5, tile_y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw all matched tile locations
        for mx, my in matches:
            cv2.rectangle(debug_img, (mx, my), (mx + tw, my + th), (255, 0, 255), 1)
        
        cv2.imwrite("debug_calibration.png", debug_img)
        logging.info(f"Found {len(matches)} closed tiles. Top-left tile at window ({tile_x}, {tile_y}), Screen origin ({origin_x}, {origin_y})")
        
        return {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "cell_w": tw,
            "cell_h": th
        }

    def analyze_board(self, board_info):
        screen = self.get_screenshot()
        rows = board_info.get('rows', 9)
        cols = board_info.get('cols', 9)
        
        matrix = np.full((rows, cols), -1, dtype=int)
        
        rel_origin_x = board_info['origin_x'] - self.window_offset_x
        rel_origin_y = board_info['origin_y'] - self.window_offset_y
        
        for r in range(rows):
            for c in range(cols):
                x = rel_origin_x + (c * board_info['cell_w']) - board_info['cell_w'] // 2
                y = rel_origin_y + (r * board_info['cell_h']) - board_info['cell_h'] // 2
                
                # Skip cells that fall outside the captured window area
                if (x < 0 or y < 0 or
                    x + board_info['cell_w'] > screen.shape[1] or
                    y + board_info['cell_h'] > screen.shape[0]):
                    logging.warning(f"Cell ({r},{c}) at crop ({x},{y}) outside screenshot bounds ({screen.shape[1]}x{screen.shape[0]}). Marking as closed.")
                    matrix[r, c] = -1
                    continue
                
                cell_img = screen[y : y + board_info['cell_h'], x : x + board_info['cell_w']]
                
                if "closed_tile.png" in self.templates:
                    res_closed = cv2.matchTemplate(cell_img, self.templates["closed_tile.png"], cv2.TM_CCOEFF_NORMED)
                    _, max_closed, _, _ = cv2.minMaxLoc(res_closed)
                    if max_closed > 0.9:
                        matrix[r, c] = -1
                        continue
                
                matched = False
                for name, template in self.templates.items():
                    if not name.endswith('.png') or name == "closed_tile.png":
                        continue
                    
                    res = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    
                    if max_val > 0.85:
                        val = self._map_template_to_value(name)
                        matrix[r, c] = val
                        matched = True
                        break
                
                if not matched:
                    matrix[r, c] = -2
        
        return matrix

    def _map_template_to_value(self, name):
        name = name.lower()
        if name == 'closed_tile.png': return -1
        if name in ('open_blank.png', 'blank.png'): return -2
        if name == 'mine.png': return 9
        if name == 'flag.png': return 10
        try:
            return int(''.join(filter(str.isdigit, name)))
        except ValueError:
            return -2
