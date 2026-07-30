import pyautogui
import logging
from src.timer import timer

class Controller:
    def __init__(self, origin_x, origin_y, cell_w, cell_h):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cell_w = cell_w
        self.cell_h = cell_h

    @timer
    def click_cell(self, row, col, right_click=False):
        target_x = self.origin_x + (col * self.cell_w)
        target_y = self.origin_y + (row * self.cell_h)
        button = 'right' if right_click else 'left'
        logging.info(f"Executing {button} click at row {row}, col {col} -> Screen({target_x}, {target_y})")
        pyautogui.moveTo(target_x, target_y)
        pyautogui.click(button=button)

    @timer
    def click_screen_pos(self, x, y, right_click=False):
        button = 'right' if right_click else 'left'
        pyautogui.click(x, y, button=button)
