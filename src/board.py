import cv2
import numpy as np
import logging

class Board:
    def __init__(self, capture, matcher, expected_cols=30):
        self.capture = capture
        self.matcher = matcher
        self.expected_cols = expected_cols
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
        bx, by, bw, bh = br
        cx1, cy1 = bx + bw // 2, by + bh // 2
        cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
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
        tile_th = 0.70
        open_th = 0.35
        MARGIN = 1  # remove 1px dark border from open cells before matching content templates

        def match_with_margin(cell_img, template, margin):
            """Match cell_img against template, both cropped to interior region."""
            if margin > 0 and template.shape[0] > 2*margin and template.shape[1] > 2*margin:
                inner = cell_img[margin:-margin, margin:-margin]
                tmpl_inner = template[margin:-margin, margin:-margin]
                res = cv2.matchTemplate(inner, tmpl_inner, cv2.TM_CCOEFF_NORMED)
            else:
                res = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            return max_val

        for r in range(rows):
            for c in range(cols):
                x = int(ox - crop_w / 2 + c * board_info['cell_w'] + 0.5)
                y = int(oy - crop_h / 2 + r * board_info['cell_h'] + 0.5)
                logging.debug(f"Cell ({r},{c}) at crop ({x},{y})")
                if (x < 0 or y < 0 or
                    x + crop_w > screen.shape[1] or
                    y + crop_h > screen.shape[0]):
                    logging.warning(f"Cell ({r},{c}) at crop ({x},{y}) outside screenshot bounds "
                                    f"({screen.shape[1]}x{screen.shape[0]}). Marking as closed.")
                    matrix[r, c] = -1
                    scores[r, c] = 0.0
                    continue

                cell_img = screen[y : y + crop_h, x : x + crop_w]

                # Pick the best match among all content templates (open_blank, numbers, flag/mine)
                best_val = None
                best_score = -1.0

                # 1a. Open blank
                tmpl = self.matcher.templates.get("open_blank.png")
                if tmpl is not None:
                    open_score = match_with_margin(cell_img, tmpl, MARGIN)
                else:
                    open_score = 0.0
                logging.debug(f"open_blank.png: {open_score}")
                if open_score > open_th and open_score > best_score:
                    best_score = open_score
                    best_val = -2

                # 1b. Number templates (digit-only, no background)
                inner = cell_img[MARGIN:-MARGIN, MARGIN:-MARGIN]
                for name in ("1.png", "2.png", "3.png", "4.png", "5.png"):
                    tmpl = self.matcher.templates.get(name.replace(".png", "_digit.png"))
                    if tmpl is None:
                        continue
                    if inner.shape[0] < tmpl.shape[0] or inner.shape[1] < tmpl.shape[1]:
                        continue
                    res = cv2.matchTemplate(inner, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, score, _, _ = cv2.minMaxLoc(res)
                    if score > tile_th and score > best_score:
                        best_score = score
                        best_val = self.matcher.map_value(name)

                # 1c. Mine/flag templates
                for name, template in self.matcher.templates.items():
                    if name not in ('mine.png', 'flag.png'):
                        continue
                    if cell_img.shape[0] < template.shape[0] or cell_img.shape[1] < template.shape[1]:
                        continue
                    score = match_with_margin(cell_img, template, MARGIN)
                    if score > 0.50 and score > best_score:
                        best_score = score
                        best_val = self.matcher.map_value(name)

                if best_val is not None:
                    matrix[r, c] = best_val
                    scores[r, c] = best_score
                    continue

                # 3. Closed-tile check with adaptive threshold
                closed_score = match_with_margin(cell_img, self.matcher.templates.get("closed_tile.png"), 0)
                logging.debug(f"closed_tile.png: {closed_score}")
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

