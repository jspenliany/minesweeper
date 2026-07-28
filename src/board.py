import cv2
import numpy as np
import logging

# Known tile templates (cell values). Skip anchors, closed/open, blank etc.
TILE_TEMPLATES = {"1.png", "2.png", "3.png", "4.png", "5.png",
                  "mine.png", "flag.png"}

class Board:
    def __init__(self, capture, matcher, expected_cols=30,
                 cell_threshold=0.85):
        self.capture = capture
        self.matcher = matcher
        self.expected_cols = expected_cols
        self.cell_threshold = cell_threshold
        self.marked_cells = set()

    def find_board(self):
        """Locate board via two corner anchors.
        board_tl.png → Cell(0,0) center.
        board_br.png → Cell(last_row, last_col) center.
        Derives cols, rows, and precise step from the two points.
        """
        screen = self.capture.get_screenshot()
        debug_img = screen.copy()

        tl = self.matcher.find_image(screen, "board_tl.png")
        if not tl:
            logging.error("Could not find board_tl anchor.")
            cv2.imwrite("debug_no_anchor.png", debug_img)
            return None
        ax, ay, aw, ah = tl
        cx0, cy0 = ax + aw // 2, ay + ah // 2
        cv2.rectangle(debug_img, (ax, ay), (ax + aw, ay + ah), (0, 255, 0), 2)
        cv2.putText(debug_img, "TL", (ax, ay - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        br = self.matcher.find_image(screen, "board_br.png")
        if not br:
            logging.error("Could not find board_br anchor.")
            cv2.imwrite("debug_no_anchor.png", debug_img)
            return None
        bx, by, bw_, bh_ = br
        cx1, cy1 = bx + bw_ // 2, by + bh_ // 2
        cv2.rectangle(debug_img, (bx, by), (bx + bw_, by + bh_), (0, 255, 0), 2)
        cv2.putText(debug_img, "BR", (bx, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        step = 31.0  # exact cell-to-cell pitch, verified by TL-BR span
        cols = int(round((cx1 - cx0) / step)) + 1
        rows = int(round((cy1 - cy0) / step)) + 1

        if cols < 2 or rows < 2:
            logging.error(f"Implausible board: {cols}x{rows} (TL=({cx0},{cy0}) BR=({cx1},{cy1}))")
            return None

        logging.info(f"TL=({cx0},{cy0}) BR=({cx1},{cy1}) → {cols} cols × {rows} rows, step={step}")

        cv2.circle(debug_img, (cx0, cy0), 4, (0, 0, 255), -1)
        cv2.circle(debug_img, (cx1, cy1), 4, (0, 0, 255), -1)

        left = int(cx0 - step / 2 + 0.5)
        top = int(cy0 - step / 2 + 0.5)
        right = int(cx1 + step / 2 + 0.5)
        bottom = int(cy1 + step / 2 + 0.5)
        cv2.rectangle(debug_img, (left, top), (right, bottom), (255, 0, 255), 1)

        cv2.imwrite("debug_calibration.png", debug_img)

        origin_x, origin_y = self.capture.to_screen(cx0, cy0)
        return {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "cell_w": step,
            "cell_h": step,
            "win_ox": cx0,
            "win_oy": cy0,
            "window_offset_x": self.capture.window_offset_x,
            "window_offset_y": self.capture.window_offset_y,
            "rows": rows,
            "cols": cols,
        }

    def compute_closed_baseline(self, board_info):
        """Compute reference closed_tile match score for every cell on a fresh all-closed board.
        Returns a (rows x cols) float32 matrix. Each entry is the baseline threshold
        for that cell — used to account for board-wide color gradient.
        """
        screen = self.capture.get_screenshot()
        rows = board_info.get('rows', 9)
        cols = board_info.get('cols', 9)

        closed_tmpl = self.matcher.templates.get("closed_tile.png")
        crop_w = closed_tmpl.shape[1] if closed_tmpl is not None else int(round(board_info['cell_w']))
        crop_h = closed_tmpl.shape[0] if closed_tmpl is not None else int(round(board_info['cell_h']))

        ox = board_info.get('win_ox', board_info['origin_x'] - board_info.get('window_offset_x', 0))
        oy = board_info.get('win_oy', board_info['origin_y'] - board_info.get('window_offset_y', 0))

        baseline = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                x = int(ox - crop_w / 2 + c * board_info['cell_w'] + 0.5)
                y = int(oy - crop_h / 2 + r * board_info['cell_h'] + 0.5)
                if (x < 0 or y < 0 or
                    x + crop_w > screen.shape[1] or
                    y + crop_h > screen.shape[0]):
                    baseline[r, c] = 0.70  # safe fallback
                    continue
                cell_img = screen[y : y + crop_h, x : x + crop_w]
                baseline[r, c] = self.matcher.match_cell(cell_img, "closed_tile.png")
        return baseline

    def analyze_board(self, board_info):
        screen = self.capture.get_screenshot()
        rows = board_info.get('rows', 9)
        cols = board_info.get('cols', 9)

        closed_tmpl = self.matcher.templates.get("closed_tile.png")
        if closed_tmpl is not None:
            crop_w = closed_tmpl.shape[1]
            crop_h = closed_tmpl.shape[0]
        else:
            crop_w = crop_h = int(round(board_info['cell_w']))

        matrix = np.full((rows, cols), -1, dtype=int)
        scores = np.full((rows, cols), 0.0, dtype=float)

        ox = board_info.get('win_ox', board_info['origin_x'] - board_info.get('window_offset_x', 0))
        oy = board_info.get('win_oy', board_info['origin_y'] - board_info.get('window_offset_y', 0))

        closed_baseline = board_info.get('closed_baseline')
        tile_th = 0.25  # threshold for number templates
        open_th = 0.35

        for r in range(rows):
            for c in range(cols):
                x = int(ox - crop_w / 2 + c * board_info['cell_w'] + 0.5)
                y = int(oy - crop_h / 2 + r * board_info['cell_h'] + 0.5)
                logging.info(f"Cell ({r},{c}) at crop ({x},{y}) analyze board ")
                if (x < 0 or y < 0 or
                    x + crop_w > screen.shape[1] or
                    y + crop_h > screen.shape[0]):
                    logging.warning(f"Cell ({r},{c}) at crop ({x},{y}) outside screenshot bounds "
                                    f"({screen.shape[1]}x{screen.shape[0]}). Marking as closed.")
                    matrix[r, c] = -1
                    scores[r, c] = 0.0
                    continue

                cell_img = screen[y : y + crop_h, x : x + crop_w]

                # 1. Number templates check (1-5.png) — take best match, skip mine/flag
                matched = False
                best_val = None
                best_score = -1.0
                for name, template in self.matcher.templates.items():
                    # logging.info(f"number...")
                    if name not in ("1.png", "2.png", "3.png", "4.png", "5.png"):
                        continue
                    logging.info(f"name: {name}, template.shape: {template.shape} ")
                    if cell_img.shape[0] < template.shape[0] or cell_img.shape[1] < template.shape[1]:
                        continue
                    score = self.matcher.match_cell(cell_img, name)
                    logging.info(f"score: {score}")
                    if score > tile_th and score > best_score:
                        best_score = score
                        best_val = self.matcher.map_value(name)
                if best_val is not None:
                    matrix[r, c] = best_val
                    scores[r, c] = best_score
                    matched = True

                if matched:
                    continue

                # 1b. Mine/flag template checks (higher threshold to avoid false positives)
                for name, template in self.matcher.templates.items():
                    # logging.info(f"mine/flag....")
                    if name not in ('mine.png', 'flag.png'):
                        continue
                    logging.info(f"name: {name}, template.shape: {template.shape} ")
                    if cell_img.shape[0] < template.shape[0] or cell_img.shape[1] < template.shape[1]:
                        continue
                    score = self.matcher.match_cell(cell_img, name)
                    logging.info(f"score: {score}")
                    if score > 0.50:
                        best_val = self.matcher.map_value(name)
                        matrix[r, c] = best_val
                        scores[r, c] = score
                        matched = True
                        break

                # 2. Open blank check
                open_score = self.matcher.match_cell(cell_img, "open_blank.png")
                logging.info(f"open_blank.png: {open_score}")
                if open_score > open_th:
                    matrix[r, c] = -2
                    scores[r, c] = open_score
                    continue

                # 3. Closed-tile check with adaptive threshold
                closed_score = self.matcher.match_cell(cell_img, "closed_tile.png")
                logging.info(f"closed_tile.png: {closed_score}")
                if closed_baseline is not None:
                    ref = closed_baseline[r, c]
                    closed_th = max(ref * 0.65, ref - 0.20)
                else:
                    closed_th = 0.70
                if closed_score > closed_th:
                    matrix[r, c] = -1
                else:
                    matrix[r, c] = -3
                scores[r, c] = closed_score

        return matrix, scores

    def mark_cell(self, r, c):
        self.marked_cells.add((r, c))

    def apply_marks(self, grid):
        grid = grid.copy()
        for r, c in self.marked_cells:
            if grid[r, c] != 10 and grid[r, c] != 9:
                grid[r, c] = 10
        return grid
