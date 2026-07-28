import cv2
import numpy as np
import logging
from src.capture import Capture
from src.matcher import Matcher

class Board:
    def __init__(self, capture, matcher, expected_cols=30,
                 tile_threshold=0.7, cell_threshold=0.85, closed_threshold=0.72):
        self.capture = capture
        self.matcher = matcher
        self.expected_cols = expected_cols
        self.tile_threshold = tile_threshold
        self.cell_threshold = cell_threshold
        self.closed_threshold = closed_threshold
        self.marked_cells = set()

    def find_board(self):
        """Locate the board origin and measure cell spacing.
        Anchor template marks cell (0,0); tile matches to the right and below
        give the center-to-center step.
        Returns dict with origin_x/y (screen coords), cell_w/h, window_offset_x/y,
        or None on failure.
        """
        screen = self.capture.get_screenshot()
        debug_img = screen.copy()

        anchor = self.matcher.find_image(screen, "board_tl.png")
        if not anchor:
            logging.error("Could not find board top-left anchor.")
            cv2.imwrite("debug_no_anchor.png", debug_img)
            return None

        ax, ay, aw, ah = anchor
        cv2.rectangle(debug_img, (ax, ay), (ax + aw, ay + ah), (0, 255, 0), 2)
        cv2.putText(debug_img, "Anchor=Cell(0,0)", (ax, ay - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        logging.info(f"Anchor (cell 0,0) at window-relative ({ax}, {ay}), size ({aw}x{ah})")

        tile_template = self.matcher.templates.get("closed_tile.png")
        if tile_template is None:
            logging.error("closed_tile.png missing in assets.")
            return None
        tw, th = tile_template.shape[1], tile_template.shape[0]
        logging.info(f"Cell template size: {tw}x{th}")

        search_x = ax
        search_y = ay
        search_w = min(screen.shape[1] - search_x, 1200)
        search_h = min(screen.shape[0] - search_y, 600)
        if search_w < tw or search_h < th:
            logging.error(f"Search region too small ({search_w}x{search_h}).")
            return None

        search_region = screen[search_y : search_y + search_h, search_x : search_x + search_w]
        cv2.rectangle(debug_img, (search_x, search_y), (search_x + search_w, search_y + search_h), (255, 0, 0), 1)

        matches = self.matcher.match_all(search_region, "closed_tile.png", threshold=self.tile_threshold)
        if not matches:
            logging.error("Could not find any closed tiles near the anchor.")
            cv2.imwrite("debug_no_tile.png", debug_img)
            return None

        matches = [(search_x + x, search_y + y) for x, y in matches]

        tol = 2
        buckets = []
        for mx, my in matches:
            placed = False
            for bucket in buckets:
                if abs(bucket[0] - my) <= tol:
                    bucket[1] += 1
                    bucket[2].append(mx)
                    placed = True
                    break
            if not placed:
                buckets.append([my, 1, [mx]])

        min_per_row = max(5, self.expected_cols * 2 // 3)
        grid_rows = sorted([b for b in buckets if b[1] >= min_per_row], key=lambda b: b[0])

        if not grid_rows:
            logging.error(f"Could not find any full grid rows. {len(buckets)} buckets, {len(matches)} total matches.")
            cv2.imwrite("debug_no_tile.png", debug_img)
            return None

        tile_x, tile_y = ax, ay

        # For each row, do 1D NMS on xs to get one match per cell
        def nms_1d(xs, min_gap):
            xs = sorted(xs)
            result = []
            for x in xs:
                if not result or x - result[-1] >= min_gap:
                    result.append(x)
            return result

        row_xs = nms_1d(grid_rows[0][2], tw // 2)
        if len(row_xs) >= 2:
            span_x = row_xs[-1] - row_xs[0]
            cell_step_x = span_x / (len(row_xs) - 1)
            logging.info(f"Row0 NMS: {len(row_xs)} cells, span_x={span_x}, step_x={cell_step_x:.4f}, xs={row_xs[:3]}...{row_xs[-3:]}")
        else:
            cell_step_x = float(tw)

        if len(grid_rows) >= 2:
            row_ys = sorted([b[0] for b in grid_rows])
            span_y = row_ys[-1] - row_ys[0]
            cell_step_y = span_y / (len(row_ys) - 1)
        else:
            cell_step_y = float(th)

        logging.info(f"Computed cell step: {cell_step_x}x{cell_step_y} (template was {tw}x{th})")
        logging.info(f"Using anchor as Cell(0,0) at ({tile_x}, {tile_y})")

        board_right = int(tile_x + self.expected_cols * cell_step_x)
        board_bottom = int(tile_y + 16 * cell_step_y)
        if board_right > screen.shape[1] or board_bottom > screen.shape[0]:
            logging.warning(f"Board extends beyond screenshot: right={board_right}, bottom={board_bottom}, "
                           f"size=({screen.shape[1]}x{screen.shape[0]})")

        origin_x, origin_y = self.capture.to_screen(ax + tw // 2, ay + th // 2)

        cv2.circle(debug_img, (ax + tw // 2, ay + th // 2), 4, (0, 0, 255), -1)
        cv2.putText(debug_img, "Cell(0,0)", (ax + int(round(cell_step_x)) + 5, ay + int(round(cell_step_y // 2))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for mx, my in matches:
            cv2.rectangle(debug_img, (mx, my), (mx + tw, my + th), (255, 0, 255), 1)

        cv2.imwrite("debug_calibration.png", debug_img)
        logging.info(f"Screen origin ({origin_x}, {origin_y})")

        return {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "cell_w": cell_step_x,
            "cell_h": cell_step_y,
            "col_xs": row_xs,  # exact match x positions for first row (window-relative)
            "row_ys": [b[0] for b in grid_rows],  # exact match y positions per row
            "window_offset_x": self.capture.window_offset_x,
            "window_offset_y": self.capture.window_offset_y
        }

    def analyze_board(self, board_info):
        screen = self.capture.get_screenshot()
        rows = board_info.get('rows', 9)
        cols = board_info.get('cols', 9)

        tile_template = self.matcher.templates.get("closed_tile.png")
        crop_w = 25
        crop_h = 25

        matrix = np.full((rows, cols), -1, dtype=int)

        col_xs = board_info.get('col_xs')  # exact column positions from find_board
        row_ys = board_info.get('row_ys')

        rel_origin_y = board_info['origin_y'] - self.capture.window_offset_y

        for r in range(rows):
            # Use exact row y if available, else fall back to step-based
            if row_ys and r < len(row_ys):
                row_base = row_ys[r]
            else:
                row_base = int(round(rel_origin_y + r * board_info['cell_h']))
            for c in range(cols):
                # Use exact column x if available, else fall back to step-based
                if col_xs and c < len(col_xs):
                    col_base = col_xs[c]
                else:
                    col_base = int(round(board_info['origin_x'] - self.capture.window_offset_x + c * board_info['cell_w']))
                x = col_base
                y = row_base

                if (x < 0 or y < 0 or
                    x + crop_w > screen.shape[1] or
                    y + crop_h > screen.shape[0]):
                    logging.warning(f"Cell ({r},{c}) at crop ({x},{y}) outside screenshot bounds ({screen.shape[1]}x{screen.shape[0]}). Marking as closed.")
                    matrix[r, c] = -1
                    continue

                cell_img = screen[y : y + crop_h, x : x + crop_w]

                matched = False
                for name, template in self.matcher.templates.items():
                    if not name.endswith('.png') or name in ("closed_tile.png", "open_blank.png", "blank.png"):
                        continue
                    if cell_img.shape[0] < template.shape[0] or cell_img.shape[1] < template.shape[1]:
                        continue
                    score = self.matcher.match_cell(cell_img, name)
                    if score > self.cell_threshold:
                        val = self.matcher.map_value(name)
                        matrix[r, c] = val
                        matched = True
                        break

                if matched:
                    continue

                # Check if it's an opened blank cell
                if self.matcher.match_cell(cell_img, "open_blank.png") > 0.9:
                    matrix[r, c] = -2
                else:
                    matrix[r, c] = -1

        return matrix

    def mark_cell(self, r, c):
        self.marked_cells.add((r, c))

    def apply_marks(self, grid):
        grid = grid.copy()
        for r, c in self.marked_cells:
            if grid[r, c] != 10 and grid[r, c] != 9:
                grid[r, c] = 10
        return grid
