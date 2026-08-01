"""Analyze a Minesweeper screenshot and report which cells are mines and
which cells are safe to click.

This script treats the existing bot modules (src/board, src/matcher,
src/solver) as a read-only API. No existing code is modified. It only calls
existing primitives (find_board, Matcher.match_cell, Board's cell
classifiers, Solver.solve) and adds the flag detection that the bot's own
analyze_board does not expose.

Usage:
    python analyze_screenshot.py <image_path>

The image should be a capture of the Minesweeper window (same content the
bot captures, e.g. the window client area).
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.board import Board
from src.matcher import Matcher
from src.solver import Solver


class _StaticCapture:
    """Capture stand-in that always returns the provided image, so the
    existing Board API can analyze a given file instead of the live screen."""

    def __init__(self, img):
        self.img = img
        self.window_offset_x = 0
        self.window_offset_y = 0

    def get_screenshot(self):
        return self.img

    def to_screen(self, x, y):
        return int(x) + self.window_offset_x, int(y) + self.window_offset_y


def _cell_center(calib, r, c):
    """Screen-space pixel at the center of cell (r, c)."""
    x = int(calib['origin_x'] + c * calib['cell_w'])
    y = int(calib['origin_y'] + r * calib['cell_h'])
    return x, y


def _is_flag(cell):
    """True if the cell crop shows a classic Minesweeper flag.

    A flag is a red cloth confined to the TOP of the cell (below it only the
    thin pole remains). This is what separates it from a red digit "3",
    which has red strokes spanning nearly the full cell height.
    """
    h, w = cell.shape[:2]
    if h < 8 or w < 8:
        return False
    c = cell.astype(np.int32)
    b, g, r = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    red = (r > 110) & (r - b > 40) & (r - g > 40)
    total = int(red.sum())
    if total < 15:
        return False
    bottom_quarter = int(red[int(h * 0.75):, :].sum())
    return bottom_quarter == 0


def _classify_board(img, calib, matcher, board):
    """Build a matrix + flag set for the screenshot.

    Reuses existing primitives per cell:
      - closed detection: max(match_cell(closed_tile), match_cell(border))
      - open classification: Board._classify_cell_by_digit_templates then
        Board._classify_cell_by_color
    Flags are detected independently (the bot's analyze_board does not
    expose them) and only on cells that also look closed.
    Matrix encoding matches the solver: -1 unknown/closed, 0-8 number,
    -2 blank, -3 unrecognized.
    """
    rows, cols = calib['rows'], calib['cols']
    cell_w, cell_h = calib['cell_w'], calib['cell_h']
    crop_w = int(calib['visual_cell_w'])
    crop_h = int(calib['visual_cell_h'])
    ox = int(calib['origin_x'] - calib['window_offset_x'])
    oy = int(calib['origin_y'] - calib['window_offset_y'])

    matrix = np.full((rows, cols), -1, dtype=int)
    flags = []

    for r in range(rows):
        for c in range(cols):
            x = int(ox - crop_w / 2 + c * cell_w + 0.5)
            y = int(oy - crop_h / 2 + r * cell_h + 0.5)
            if x < 0 or y < 0 or x + crop_w > img.shape[1] or y + crop_h > img.shape[0]:
                continue
            cell_img = img[y:y + crop_h, x:x + crop_w]

            if _is_flag(cell_img):
                flags.append((r, c))
                continue

            closed_score = max(
                matcher.match_cell(cell_img, "closed_tile.png"),
                matcher.match_cell(cell_img, "border.png"),
            )
            if closed_score >= 0.6:
                continue

            num = board._classify_cell_by_digit_templates(cell_img)
            if num is None:
                num = board._classify_cell_by_color(cell_img)
            matrix[r, c] = num

    return matrix, flags


def _deduce(matrix, flags):
    """Run the solver logic. Returns (safe, mines, guess, solved).
    - safe:  list of (cell, reason) cells proven safe to click
    - mines: list of (cell, reason) cells deduced to be mines
    - guess: (cell, reason) or None if a random move is needed
    - solved: True when the board has no unknowns left"""
    rows, cols = matrix.shape
    matrix = matrix.copy()
    solver = Solver(rows, cols)
    for cell in flags:
        solver.marked_cells.add(cell)
    solver.update_grid(matrix)

    safe = []
    mines = []
    seen_safe = set()
    seen_mine = set()
    guess = None
    solved = False

    for _ in range(5000):
        actions = solver.solve()
        if not actions:
            break
        action, coords, reason = actions[0]
        if action == 'NONE':
            # The solver only returns NONE once it has no more -1 (closed)
            # cells left. Cells classified as -3 (unrecognized) are still
            # unknown, so "solved" here only means "no closed cells remain";
            # the caller checks for -3 cells before declaring a full solve.
            solved = True
            break
        if action == 'GUESS':
            guess = (coords, reason)
            break
        if action == 'MARK':
            if coords in solver.marked_cells:
                break
            solver.marked_cells.add(coords)
            solver.update_grid(matrix)
            if coords not in seen_mine:
                seen_mine.add(coords)
                mines.append((coords, reason))
        else:  # CLICK batch
            for _, cell, why in actions:
                # A cell proven safe is no longer unknown: exclude it from the
                # solver's unknowns so later rounds can keep propagating (e.g.
                # a number left with a single remaining unknown then marks it).
                matrix[cell] = -2
                solver.update_grid(matrix)
                if cell not in seen_safe:
                    seen_safe.add(cell)
                    safe.append((cell, why))

    return safe, mines, guess, solved


def _render_analysis(img, calib, flags, mines, safe):
    """Return a copy of the screenshot with the analysis drawn on it:
    red circles on mines deduced by the solver from closed tiles (player
    flags are already confirmed mines, so they get no circle) and green
    circles on every confirmed-safe cell to click. A small legend is drawn
    at the top-left (ASCII text, since OpenCV can't render Chinese)."""
    out = img.copy()
    radius = max(4, int(calib['cell_w'] * 0.42))
    thickness = max(2, int(calib['cell_w'] * 0.09))
    color_mine = (0, 0, 255)     # BGR red
    color_safe = (0, 255, 0)     # BGR green

    for (r, c), _ in mines:
        px, py = _cell_center(calib, r, c)
        cv2.circle(out, (px, py), radius, color_mine, thickness)
    for (r, c), _ in safe:
        px, py = _cell_center(calib, r, c)
        cv2.circle(out, (px, py), radius, color_safe, thickness)

    cv2.rectangle(out, (8, 8), (200, 62), (0, 0, 0), -1)
    cv2.circle(out, (22, 22), 6, color_mine, 2)
    cv2.putText(out, "Deduced mine", (36, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(out, (22, 46), 6, color_safe, 2)
    cv2.putText(out, "Safe click", (36, 51), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _render_grid(matrix, flags, mines):
    flagged = set(flags) | set(cell for cell, _ in mines)
    rows, cols = matrix.shape
    lines = []
    for r in range(rows):
        row = []
        for c in range(cols):
            v = matrix[r, c]
            if (r, c) in flagged:
                row.append('F')
            elif v == -1:
                row.append('?')
            elif v == -2:
                row.append(' ')
            elif v == -3:
                row.append('?')
            elif v >= 0:
                row.append(str(v))
            else:
                row.append('?')
        lines.append(''.join(row))
    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_screenshot.py <图片路径>")
        return 1

    # UTF-8 output so Chinese renders correctly in modern terminals.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    path = os.path.abspath(sys.argv[1])
    img = cv2.imread(path)
    if img is None:
        print(f"无法读取图片: {path}")
        return 1

    # Run in a temp cwd so the existing find_board() debug artifact
    # (debug_calibration.png) does not pollute the project folder.
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            matcher = Matcher()
            board = Board(_StaticCapture(img), matcher)
            calib = board.find_board()
        finally:
            os.chdir(old_cwd)

    if not calib:
        print("未能定位棋盘（锚点/轮廓/颜色检测均失败）。请确认截图是扫雷窗口。")
        return 1

    matrix, flags = _classify_board(img, calib, matcher, board)
    safe, mines, guess, solved = _deduce(matrix, flags)

    rows, cols = calib['rows'], calib['cols']
    anchored = calib.get('anchor_matched', False)

    out = _render_analysis(img, calib, flags, mines, safe)
    out_path = os.path.splitext(path)[0] + "_analysis.png"
    if not cv2.imwrite(out_path, out):
        print(f"无法写入分析图: {out_path}")
        return 1

    unrecognized = int((matrix == -3).sum())
    closed_left = int((matrix == -1).sum())

    print("=" * 60)
    print("===== 扫雷截图分析 =====")
    print("=" * 60)
    print(f"图片: {path} ({img.shape[1]}x{img.shape[0]})")
    print(f"棋盘: {cols}列 x {rows}行 | step={calib['cell_w']:.1f} | "
          f"单元格 {calib['visual_cell_w']}x{calib['visual_cell_h']}")
    print(f"锚点匹配: {'是' if anchored else '否 (精度可能较低)'}")
    print(f"分析图已保存: {out_path}")
    print()
    print("图例: 红圈 = 推理出的雷 (closed_tile, solver 推导) | 绿圈 = 安全可点击")
    print("      (已有红旗的格子不标圈)")
    print()
    print("--- 结论 ---")
    if solved and unrecognized == 0:
        print("  棋盘已全部解出 (剩余未知格为 0)。")
    else:
        print(f"  红旗 (玩家已确认雷): {len(flags)} 个 (不标圈)")
        print(f"  推理出的雷 (closed_tile): {len(mines)} 个  -> 红圈")
        print(f"  安全可点: {len(safe)} 个  -> 绿圈")
        if guess is not None:
            r, c = guess[0]
            px, py = _cell_center(calib, r, c)
            print(f"  无法确定时可选: 随机猜 ({r}, {c})  中心像素 ({px},{py})")
        if unrecognized > 0:
            print(f"  注意: 有 {unrecognized} 个格子未能识别(开格未分类)，"
                  f"{closed_left} 个闭格未知，结果不完整")

    print()
    print("--- 棋盘状态图 (F=雷/?=未知/空格=空/数字=周围雷数) ---")
    print(_render_grid(matrix, flags, mines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
