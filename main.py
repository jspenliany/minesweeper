import time
import logging
import pyautogui
import cv2
import ctypes
from src.capture import Capture, find_window_by_title
from src.matcher import Matcher
from src.board import Board
from src.solver import Solver, Reasoner
from src.controller import Controller

user32 = ctypes.windll.user32

def get_foreground_window_title():
    """Get the title of the currently active (foreground) window"""
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = user32.GetWindowTextLengthW(hwnd) + 1
    buffer = ctypes.create_unicode_buffer(length)
    user32.GetWindowTextW(hwnd, buffer, length)
    return buffer.value

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("minesweeper_bot.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class MinesweeperBot:
    def __init__(self):
        self.capture = Capture()
        self.matcher = Matcher()
        self.board = Board(self.capture, self.matcher)
        self.controller = None
        self.solver = None
        self.state = "IDLE"
        self.rows = None
        self.cols = None
        self.flag_screenshot_counter = 0
        self.calib = None  # latest board calibration data

    def run(self):
        logging.info("Minesweeper Bot started. Waiting for game...")
        while True:
            if self.state == "IDLE":
                self._handle_idle()
            elif self.state == "START_GAME":
                self._handle_start_game()
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
        screen = self.capture.get_screenshot()
        anchor = self.matcher.find_image(screen, "board_tl.png")
        if anchor:
            ax, ay, aw, ah = anchor
            sx = ax + aw // 2 + self.capture.window_offset_x
            sy = ay + ah // 2 + self.capture.window_offset_y
            pyautogui.click(sx, sy)
            time.sleep(0.2)
            logging.info("Focused game window via anchor click.")
            return True
        return False

    def _handle_idle(self):
        logging.info("State: IDLE. Searching for game window...")
        screen = self.capture.get_screenshot()
        if self.matcher.find_image(screen, "board_tl.png"):
            logging.info("Game window detected!")
            self.state = "START_GAME"
            time.sleep(3)  # Pause before F2 to avoid rapid loop
        else:
            logging.info("Game not found. Still waiting...")

    def _handle_start_game(self):
        logging.info("State: START_GAME. Starting new game via F2...")
        try:
            self._focus_game_window()
            pyautogui.press('f2')
            # F2 during an active game triggers a "新游戏" confirmation dialog
            for _ in range(10):
                time.sleep(0.1)
                title = get_foreground_window_title()
                if "新游戏" in title:
                    logging.info("New Game dialog detected after F2. Confirming with Alt+N...")
                    pyautogui.hotkey('alt', 'n')
                    time.sleep(0.3)
                    break
            logging.info("Pressed F2. Moving to Calibration Test state.")
            self.state = "TEST_CALIBRATION"
        except Exception as e:
            logging.error(f"Hotkeys failed: {e}")
            self.state = "IDLE"

    def _save_calibration_preview(self, calib):
        """Save an annotated screenshot with corner circles (non-destructive).
        Uses step-based positions; no NMS data required.
        """
        self._focus_game_window()
        time.sleep(0.3)
        final_screen = self.capture.get_screenshot()
        step = int(round(calib['cell_w']))
        calib_rows = calib.get('rows', 16)
        calib_cols = calib.get('cols', 30)
        logging.info(f"Preview: win_ox={calib['win_ox']}, win_oy={calib['win_oy']}, "
                     f"step={step}, "
                     f"n_cols={calib_cols}, n_rows={calib_rows}")
        mid_row_y = int(calib['win_oy'] + (calib_rows // 2) * step + 0.5)
        for c in range(calib_cols):
            sx = int(calib['win_ox'] + c * step + 0.5)
            cv2.line(final_screen, (sx, mid_row_y - 10), (sx, mid_row_y + 10), (0, 255, 0), 1)
        corners = [
            (0, 0, "TL"),
            (0, calib_cols - 1, "TR"),
            (calib_rows - 1, 0, "BL"),
            (calib_rows - 1, calib_cols - 1, "BR"),
        ]
        for r, c, name in corners:
            sx = int(calib['win_ox'] + c * step + 0.5)
            sy = int(calib['win_oy'] + r * step + 0.5)
            cv2.circle(final_screen, (sx, sy), 8, (0, 0, 255), 2)
            cv2.putText(final_screen, f"{name}({sx},{sy})", (sx + 10, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite("calibration_test_final.png", final_screen)
        logging.info("Saved calibration_test_final.png with diagnostic markers.")

    def _handle_test_calibration(self):
        logging.info("State: TEST_CALIBRATION. Detecting board grid...")
        calib = self.board.find_board()
        if not calib:
            logging.error("Calibration failed (grid not found). Waiting 6s before retry...")
            time.sleep(6)
            self.state = "IDLE"
            return

        self.rows = calib.get('rows', self.rows)
        self.cols = calib.get('cols', self.cols)
        self._save_calibration_preview(calib)
        logging.info("Calibration OK. Pressing F2 to start fresh game...")
        self._focus_game_window()
        pyautogui.press('f2')
        time.sleep(1)
        # F2 during an active game triggers a "新游戏" confirmation dialog
        if "新游戏" in get_foreground_window_title():
            logging.info("New Game dialog detected after F2. Confirming with Alt+N...")
            pyautogui.hotkey('alt', 'n')
            time.sleep(0.5)
        self.state = "FIRST_MOVE"

    def _handle_first_move(self):
        logging.info("State: FIRST_MOVE. Calibrating fresh board...")
        calib = self.board.find_board()
        if not calib:
            logging.error("Calibration failed in FIRST_MOVE. Returning to IDLE.")
            self.state = "IDLE"
            return
        self.calib = calib
        self.closed_baseline = None
        self.controller = Controller(calib['origin_x'], calib['origin_y'], calib['cell_w'], calib['cell_h'])
        self.solver = Solver(self.rows, self.cols, marked_cells=self.board.marked_cells)

        # Compute per-cell closed_tile baseline from the fresh all-closed board
        board_info = {
            'origin_x': calib['origin_x'],
            'origin_y': calib['origin_y'],
            'cell_w': calib['cell_w'],
            'cell_h': calib['cell_h'],
            'rows': self.rows,
            'cols': self.cols,
            'win_ox': calib.get('win_ox'),
            'win_oy': calib.get('win_oy'),
        }
        self.closed_baseline = self.board.compute_closed_baseline(board_info)
        logging.info(f"Closed baseline range: {self.closed_baseline.min():.2f} ~ {self.closed_baseline.max():.2f}")

        # Log initial board state (pre-first-click) to verify no false positives
        board_info['closed_baseline'] = self.closed_baseline
        init_matrix, init_scores = self.board.analyze_board(board_info)
        logging.info("--- Initial Board (pre-first-move) ---")
        for r in range(self.rows):
            cells = []
            for c in range(self.cols):
                cells.append(f"({init_matrix[r,c]},{init_scores[r,c]*100:.0f}%)")
            logging.info("  " + " ".join(cells))
        logging.info("--------------------------------------")

        r, c = self.rows // 2, self.cols // 2
        self._focus_game_window()
        self.controller.click_cell(r, c)
        logging.info(f"First move at ({r}, {c}). Waiting 3 seconds before entering PLAYING state.")
        time.sleep(3)
        self.state = "PLAYING"

    def _handle_dialogs(self):
        """Check foreground window title and handle any dialogs. Returns True if a dialog was handled."""
        title = get_foreground_window_title()
        if not title:
            return False
        
        # If the main game window is active, nothing to handle
        if any(kw in title for kw in ["扫雷", "Minesweeper", "minesweeper"]):
            return False
        
        # Game over dialog
        if "游戏失败" in title or "Game Over" in title:
            logging.info("Game over dialog detected! Pressing Alt+P to start new game.")
            hwnd = user32.GetForegroundWindow()
            user32.SetForegroundWindow(hwnd)
            time.sleep(2)
            pyautogui.hotkey('alt', 'p')
            time.sleep(0.5)
            self.state = "FIRST_MOVE"
            return True

        # Victory dialog
        if "游戏胜利" in title or "Victory" in title or "You Win" in title:
            logging.info("Victory dialog detected! Pressing Alt+P to start new game.")
            hwnd = user32.GetForegroundWindow()
            user32.SetForegroundWindow(hwnd)
            time.sleep(2)
            pyautogui.hotkey('alt', 'p')
            time.sleep(0.5)
            self.state = "FIRST_MOVE"
            return True
        
        # New Game dialog
        if "新游戏" in title:
            logging.info("New Game dialog detected! Pressing Alt+K to return to game.")
            time.sleep(1)
            pyautogui.hotkey('alt', 'k')
            time.sleep(0.5)
            return True
        
        return False

    def _handle_playing(self):
        # Check for dialogs at the start of every cycle
        if self._handle_dialogs():
            return
        
        logging.info("State: PLAYING. Analyzing board...")
        board_info = {
            'origin_x': self.controller.origin_x,
            'origin_y': self.controller.origin_y,
            'cell_w': self.controller.cell_w,
            'cell_h': self.controller.cell_h,
            'rows': self.rows,
            'cols': self.cols,
            'win_ox': self.calib.get('win_ox') if self.calib else None,
            'win_oy': self.calib.get('win_oy') if self.calib else None,
            'closed_baseline': self.closed_baseline,
        }
        matrix, scores = self.board.analyze_board(board_info)
        
        logging.info("--- Current Logical Board ---")
    #---------------------------print matrix-------start
        # for r in range(self.rows):
        #     cells = []
        #     for c in range(self.cols):
        #         v = matrix[r, c]
        #         s = scores[r, c]
        #         cells.append(f"({v},{s*100:.0f}%)")
        #     logging.info("  " + " ".join(cells))
        # logging.info("--------------------------------")
    #---------------------------print matrix-------end

        self.solver.update_grid(matrix)
        action, coords, reason = self.solver.solve()
        
        if action == 'NONE':
            logging.info("Solver: " + reason)
            self.state = "RESULT"
            return

        self._focus_game_window()
        if action == 'CLICK':
            logging.info(Reasoner.format(action, coords, reason))
            self.controller.click_cell(coords[0], coords[1], right_click=False)
            time.sleep(2)
            if self._handle_dialogs(): return
        elif action == 'MARK':
            r, c = coords
            # 1. Screenshot with red circle BEFORE marking (documents the decision)
            self.flag_screenshot_counter += 1
        #-----------------------------------------------------image save-------start
            pre_img = self.capture.get_screenshot()
            cx = int(round(self.controller.origin_x + c * self.controller.cell_w - self.capture.window_offset_x))
            cy = int(round(self.controller.origin_y + r * self.controller.cell_h - self.capture.window_offset_y))
            cv2.circle(pre_img, (cx, cy), 10, (0, 0, 255), 2)
            filename = f"flag{self.flag_screenshot_counter:04d}.png"
            cv2.imwrite(filename, pre_img)
            logging.info(f"Saved {filename} with cell ({r},{c}) circled before marking.")
        #-----------------------------------------------------image save-------end
            # 2. Wait 5 seconds for user to review the screenshot
            logging.info("Waiting 2 seconds for user review...")
            time.sleep(2)
            # 3. Log reasoning
            logging.info(Reasoner.format(action, coords, reason))
            # 4. Right-click to place flag
            self.controller.click_cell(r, c, right_click=True)
            self.board.mark_cell(r, c)
            time.sleep(1)
            if self._handle_dialogs(): return
            # 5. Post-mark cascade
            self._cascade_flag_clicks()
        elif action == 'GUESS':
            logging.info(Reasoner.format(action, coords, reason))
            self.controller.click_cell(coords[0], coords[1], right_click=False)
            time.sleep(5)
            if self._handle_dialogs(): return
        
        screen = self.capture.get_screenshot()
        if self.matcher.find_image(screen, "game_over_fragment.png") or self.matcher.find_image(screen, "win_fragment.png"):
            logging.info("Game end detected!")
            self.state = "RESULT"

    def _cascade_flag_clicks(self, max_iter=100):
        """After marking a flag, immediately cascade: if a number cell's flags match its value, click all safe neighbors"""
        for _ in range(max_iter):
            if self._handle_dialogs():
                return
            board_info = {
                'origin_x': self.controller.origin_x,
                'origin_y': self.controller.origin_y,
                'cell_w': self.controller.cell_w,
                'cell_h': self.controller.cell_h,
                'rows': self.rows,
                'cols': self.cols,
                'win_ox': self.calib.get('win_ox') if self.calib else None,
                'win_oy': self.calib.get('win_oy') if self.calib else None,
                'closed_baseline': self.closed_baseline,
            }
            matrix, _ = self.board.analyze_board(board_info)
            self.solver.update_grid(matrix)
            action, coords, reason = self.solver.solve()
            if action == 'CLICK':
                logging.info(f"[Cascade] {Reasoner.format(action, coords, reason)}")
                self._focus_game_window()
                self.controller.click_cell(coords[0], coords[1], right_click=False)
                time.sleep(2)
                if self._handle_dialogs():
                    return
            elif action == 'MARK':
                r, c = coords
                self.flag_screenshot_counter += 1
        #-----------------------------------------------------image save-------start
                pre_img = self.capture.get_screenshot()
                cx = int(round(self.controller.origin_x + c * self.controller.cell_w - self.capture.window_offset_x))
                cy = int(round(self.controller.origin_y + r * self.controller.cell_h - self.capture.window_offset_y))
                cv2.circle(pre_img, (cx, cy), 10, (0, 0, 255), 2)
                filename = f"flag{self.flag_screenshot_counter:04d}.png"
                cv2.imwrite(filename, pre_img)
                logging.info(f"[Cascade] Saved {filename} with cell ({r},{c}) circled before marking.")
        #-----------------------------------------------------image save-------end
                logging.info(f"[Cascade] {Reasoner.format(action, coords, reason)}")
                self._focus_game_window()
                self.controller.click_cell(r, c, right_click=True)
                self.board.mark_cell(r, c)
                time.sleep(1)
                if self._handle_dialogs():
                    return
            else:
                break

    def _handle_result(self):
        logging.info("State: RESULT. Looking for dialogs...")
        
        # Try dialog handling first
        if self._handle_dialogs():
            return
        
        # Fallback: try to find restart button in screenshot
        screen = self.capture.get_screenshot()
        restart_btn = self.matcher.find_image(screen, "restart_button.png")
        
        if restart_btn:
            self._focus_game_window()
            rx, ry, rw, rh = restart_btn
            sx = rx + rw // 2 + self.capture.window_offset_x
            sy = ry + rh // 2 + self.capture.window_offset_y
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
