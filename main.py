import time
import logging
import pyautogui
import cv2
import numpy as np
import ctypes
from ctypes import wintypes
from src.vision import Vision, find_window_by_title
from src.solver import Solver
from src.controller import Controller

user32 = ctypes.windll.user32

def focus_minesweeper_window():
    """Find and bring the Minesweeper window to the foreground"""
    hwnd = find_window_by_title(["扫雷", "Minesweeper", "minesweeper", "扫雷游戏"])
    if hwnd:
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.2)
        logging.info("Focused Minesweeper window via Win32 API.")
        return True
    logging.warning("Minesweeper window not found by title.")
    return False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MinesweeperBot:
    def __init__(self):
        self.vision = Vision()
        self.controller = None
        self.solver = None
        self.state = "IDLE"
        self.rows = 16
        self.cols = 30

    def run(self):
        logging.info("Minesweeper Bot started. Waiting for game...")
        while True:
            if self.state == "IDLE":
                self._handle_idle()
            elif self.state == "MAIN_MENU":
                self._handle_main_menu()
            elif self.state == "TEST_CALIBRATION":
                self._handle_test_calibration()
            elif self.state == "FIRST_MOVE":
                self._handle_first_move()
            elif self.state == "PLAYING":
                self._handle_playing()
            elif self.state == "RESULT":
                self._handle_result()
            elif self.state == "EXIT":
                logging.info("Exiting program.")
                break
            time.sleep(1)

    def _focus_game_window(self):
        """Bring the Minesweeper window to focus via Win32 API or fallback click"""
        if focus_minesweeper_window():
            return True
        # Fallback: click on anchor point
        screen = self.vision.get_screenshot()
        anchor = self.vision.find_image(screen, "board_tl.png")
        if anchor:
            ax, ay, aw, ah = anchor
            sx = ax + aw // 2 + self.vision.window_offset_x
            sy = ay + ah // 2 + self.vision.window_offset_y
            pyautogui.click(sx, sy)
            time.sleep(0.2)
            logging.info("Focused game window via anchor click.")
            return True
        return False

    def _handle_idle(self):
        logging.info("State: IDLE. Searching for game window...")
        screen = self.vision.get_screenshot()
        if self.vision.find_image(screen, "board_tl.png"):
            logging.info("Game window detected!")
            self.state = "MAIN_MENU"
        else:
            logging.info("Game not found. Still waiting...")

    def _handle_main_menu(self):
        logging.info("State: MAIN_MENU. Sending hotkeys to start new game...")
        try:
            self._focus_game_window()
            pyautogui.hotkey('alt', 'g')
            time.sleep(0.3)
            pyautogui.press('f2')
            time.sleep(0.1)
            pyautogui.press('alt')  # Dismiss the menu
            logging.info("Sent Alt+G, F2, Alt. Moving to Calibration Test state.")
            self.state = "TEST_CALIBRATION"
        except Exception as e:
            logging.error(f"Hotkeys failed: {e}")
            self.state = "IDLE"

    def _handle_test_calibration(self):
        logging.info("State: TEST_CALIBRATION. Starting a trial game to verify coordinates...")
        calib = self.vision.calibrate_grid()
        if not calib:
            logging.error("Calibration failed. Returning to IDLE.")
            self.state = "IDLE"
            return

        self.controller = Controller(calib['origin_x'], calib['origin_y'], calib['cell_w'], calib['cell_h'])

        corners = [
            (0, 0, "Top-Left"),
            (0, self.cols - 1, "Top-Right"),
            (self.rows - 1, 0, "Bottom-Left"),
            (self.rows - 1, self.cols - 1, "Bottom-Right")
        ]
        
        logging.info("Clicking the 4 corner cells to verify calibration...")
        for r, c, name in corners:
            self._focus_game_window()
            self.controller.click_cell(r, c)
            time.sleep(0.5)

        # Save annotated screenshot showing where we clicked (before analysis, always)
        self._focus_game_window()
        time.sleep(0.3)
        final_screen = self.vision.get_screenshot()
        rel_ox = calib['origin_x'] - self.vision.window_offset_x
        rel_oy = calib['origin_y'] - self.vision.window_offset_y
        for r, c, name in corners:
            cx = int(rel_ox + c * calib['cell_w'])
            cy = int(rel_oy + r * calib['cell_h'])
            cv2.circle(final_screen, (cx, cy), 8, (0, 0, 255), 2)
            cv2.putText(final_screen, name, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite("calibration_test_final.png", final_screen)
        logging.info("Saved 'calibration_test_final.png' with annotated click positions.")

        board_info = {
            'origin_x': calib['origin_x'],
            'origin_y': calib['origin_y'],
            'cell_w': calib['cell_w'],
            'cell_h': calib['cell_h'],
            'rows': self.rows,
            'cols': self.cols
        }
        matrix = self.vision.analyze_board(board_info)

        all_open = True
        for r, c, name in corners:
            val = matrix[r, c]
            if val == -1:
                logging.error(f"Calibration FAILED at {name} corner ({r},{c}): cell is still closed.")
                all_open = False
            else:
                logging.info(f"Calibration OK at {name} corner ({r},{c}): value = {val}")

        if not all_open:
            logging.error("Calibration failed. Returning to IDLE.")
            self.state = "IDLE"
            return

        logging.info("All 4 corners opened! Coordinates correct. Starting a real game...")
        # Start a fresh game (the test board has 4 opened cells)
        self._focus_game_window()
        pyautogui.hotkey('alt', 'g')
        time.sleep(0.3)
        pyautogui.press('f2')
        time.sleep(0.1)
        pyautogui.press('alt')
        time.sleep(0.5)
        self.state = "FIRST_MOVE"

    def _handle_first_move(self):
        logging.info("State: FIRST_MOVE. Calibrating fresh board...")
        calib = self.vision.calibrate_grid()
        if not calib:
            logging.error("Calibration failed in FIRST_MOVE. Returning to IDLE.")
            self.state = "IDLE"
            return
        self.controller = Controller(calib['origin_x'], calib['origin_y'], calib['cell_w'], calib['cell_h'])
        self.solver = Solver(self.rows, self.cols)
        r, c = self.rows // 2, self.cols // 2
        self._focus_game_window()
        self.controller.click_cell(r, c)
        logging.info(f"First move at ({r}, {c}). Entering PLAYING state.")
        self.state = "PLAYING"

    def _handle_playing(self):
        # Check for game over dialog before analyzing
        if find_window_by_title(["游戏失败", "Game Over"]):
            logging.info("Game over dialog detected!") 
            self.state = "RESULT"
            return
        
        logging.info("State: PLAYING. Analyzing board...")
        board_info = {
            'origin_x': self.controller.origin_x,
            'origin_y': self.controller.origin_y,
            'cell_w': self.controller.cell_w,
            'cell_h': self.controller.cell_h,
            'rows': self.rows,
            'cols': self.cols
        }
        matrix = self.vision.analyze_board(board_info)
        
        print("\n--- Current Logical Board ---")
        # Only print first 5 rows to avoid console flooding
        print(matrix[:5])
        print("-----------------------------\n")
        
        self.solver.update_grid(matrix)
        action, coords, reason = self.solver.solve()
        
        if action == 'NONE':
            logging.info("Solver: " + reason)
            self.state = "RESULT"
            return

        self._focus_game_window()
        if action == 'CLICK':
            logging.info(self.solver.get_reasoning(action, coords, reason))
            self.controller.click_cell(coords[0], coords[1], right_click=False)
            time.sleep(0.3)
        elif action == 'MARK':
            logging.info(self.solver.get_reasoning(action, coords, reason))
            self.controller.click_cell(coords[0], coords[1], right_click=True)
            time.sleep(0.3)
        elif action == 'GUESS':
            logging.info(self.solver.get_reasoning(action, coords, reason))
            self.controller.click_cell(coords[0], coords[1], right_click=False)
            time.sleep(0.3)

        screen = self.vision.get_screenshot()
        if self.vision.find_image(screen, "game_over_fragment.png") or self.vision.find_image(screen, "win_fragment.png"):
            logging.info("Game end detected!")
            self.state = "RESULT"

    def _handle_result(self):
        logging.info("State: RESULT. Game over. Looking for dialog...")
        
        # Try to handle "游戏失败" dialog with keyboard shortcuts
        dialog_hwnd = find_window_by_title(["游戏失败", "Game Over"])
        if dialog_hwnd:
            user32.SetForegroundWindow(dialog_hwnd)
            time.sleep(0.2)
            pyautogui.press('p')  # P = new game
            logging.info("Pressed P to start a new game.")
            time.sleep(0.5)
            self.state = "FIRST_MOVE"
            return
        
        # Fallback: try to find restart button in screenshot
        screen = self.vision.get_screenshot()
        restart_btn = self.vision.find_image(screen, "restart_button.png")
        
        if restart_btn:
            self._focus_game_window()
            rx, ry, rw, rh = restart_btn
            sx = rx + rw // 2 + self.vision.window_offset_x
            sy = ry + rh // 2 + self.vision.window_offset_y
            self.controller.click_screen_pos(sx, sy)
            logging.info("Clicked 'Restart' button. Starting next game.")
            time.sleep(0.5)
            self.state = "FIRST_MOVE"
        else:
            logging.warning("No dialog or restart button found. Returning to IDLE.")
            self.state = "IDLE"

if __name__ == "__main__":
    bot = MinesweeperBot()
    bot.run()
