import time
import logging
import pyautogui
import cv2
import numpy as np
from src.vision import Vision
from src.solver import Solver
from src.controller import Controller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MinesweeperBot:
    def __init__(self):
        self.vision = Vision()
        self.controller = None
        self.solver = None
        self.state = "IDLE"
        self.rows = 16  # Updated for 30x16 board
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
            time.sleep(1)

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
            pyautogui.hotkey('alt', 'g')
            time.sleep(0.3)
            pyautogui.press('f2')
            logging.info("Sent Alt+G and F2. Moving to Calibration Test state.")
            self.state = "TEST_CALIBRATION"
        except Exception as e:
            logging.error(f"Hotkeys failed: {e}")
            self.state = "IDLE"

    def _handle_test_calibration(self):
        logging.info("State: TEST_CALIBRATION. Validating coordinate mapping...")
        calib = self.vision.calibrate_grid()
        if not calib:
            logging.error("Calibration failed. Returning to IDLE.")
            self.state = "IDLE"
            return

        self.controller = Controller(calib['origin_x'], calib['origin_y'], calib['cell_w'], calib['cell_h'])
        
        screen = self.vision.get_screenshot()
        cv2.circle(screen, (calib['origin_x'], calib['origin_y']), 5, (0, 0, 255), -1)
        cv2.imwrite("debug_origin.png", screen)
        logging.info("Saved 'debug_origin.png'. Please check if the red dot is in the center of cell (0,0).")

        test_points = [
            (0, 0, "Top-Left"),
            (0, self.cols - 1, "Top-Right"),
            (self.rows - 1, 0, "Bottom-Left"),
            (self.rows - 1, self.cols - 1, "Bottom-Right")
        ]
        
        logging.info("Starting Coordinate Test. Please observe the mouse movement!")
        for r, c, name in test_points:
            logging.info(f"Testing {name} corner at ({r}, {c})...")
            self.controller.click_cell(r, c)
            time.sleep(1.5)
            
        logging.info("Test completed. Moving to First Move.")
        time.sleep(2)
        self.state = "FIRST_MOVE"

    def _handle_first_move(self):
        logging.info("State: FIRST_MOVE. Executing initial random move...")
        self.solver = Solver(self.rows, self.cols)
        r, c = self.rows // 2, self.cols // 2
        logging.info(f"Performing first move at ({r}, {c})")
        self.controller.click_cell(r, c)
        self.state = "PLAYING"

    def _handle_playing(self):
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
        print(matrix)
        print("-----------------------------\n")
        
        self.solver.update_grid(matrix)
        action, coords, reason = self.solver.solve()
        
        if action == 'NONE':
            logging.info("Solver: " + reason)
            self.state = "RESULT"
            return

        if action == 'CLICK':
            logging.info(self.solver.get_reasoning(action, coords, reason))
            self.controller.click_cell(coords[0], coords[1], right_click=False)
            time.sleep(0.5)
        elif action == 'MARK':
            logging.info(self.solver.get_reasoning(action, coords, reason))
            self.controller.click_cell(coords[0], coords[1], right_click=True)
            time.sleep(0.5)
        elif action == 'GUESS':
            logging.info(self.solver.get_reasoning(action, coords, reason))
            self.controller.click_cell(coords[0], coords[1], right_click=False)
            time.sleep(0.5)

        screen = self.vision.get_screenshot()
        if self.vision.find_image(screen, "game_over_fragment.png") or self.vision.find_image(screen, "win_fragment.png"):
            logging.info("Game end detected via fragment!")
            self.state = "RESULT"

    def _handle_result(self):
        logging.info("State: RESULT. Cleaning up and restarting...")
        screen = self.vision.get_screenshot()
        restart_btn = self.vision.find_image(screen, "restart_button.png")
        
        if restart_btn:
            rx, ry, rw, rh = restart_btn
            self.controller.click_screen_pos(rx + rw // 2, ry + rh // 2)
            logging.info("Clicked 'Restart'. Returning to Main Menu.")
            self.state = "MAIN_MENU"
        else:
            logging.warning("Could not find restart button. Returning to Idle.")
            self.state = "IDLE"

if __name__ == "__main__":
    bot = MinesweeperBot()
    bot.run()
