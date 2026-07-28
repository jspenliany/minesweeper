import pyautogui
import logging

class Controller:
    def __init__(self, origin_x, origin_y, cell_w, cell_h):
        """
        :param origin_x: 棋盘左上角第一个单元格的中心 X 坐标
        :param origin_y: 棋盘左上角第一个单元格的中心 Y 坐标
        :param cell_w: 单个单元格的宽度
        :param cell_h: 单个单元格的高度
        :param cell_w: 单个单元格的宽度
        :param cell_h: 单个单元格的高度
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cell_w = cell_w
        self.cell_h = cell_h

    def click_cell(self, row, col, right_click=False):
        """
        执行点击操作
        :param row: 逻辑行 (0-indexed)
        :param col: 逻辑列 (0-indexed)
        :param right_click: 是否右键点击 (标记旗帜)
        """
        target_x = self.origin_x + (col * self.cell_w)
        target_y = self.origin_y + (row * self.cell_h)
        
        button = 'right' if right_click else 'left'
        logging.info(f"Executing {button} click at row {row}, col {col} -> Screen({target_x}, {target_y})")
        
        pyautogui.moveTo(target_x, target_y)
        pyautogui.click(button=button)

    def click_screen_pos(self, x, y, right_click=False):
        """直接点击屏幕绝对坐标"""
        button = 'right' if right_click else 'left'
        pyautogui.click(x, y, button=button)
