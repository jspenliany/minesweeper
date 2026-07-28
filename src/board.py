import cv2
import numpy as np
import logging

class Board:
    def __init__(self, capture, matcher, expected_cols=30):
        self.capture = capture
        self.matcher = matcher
        self.expected_cols = expected_cols
        self.marked_cells = set()

    @staticmethod
    def _nms(matches, min_dist):
        results = []
        for m in matches:
            mx, my = m
            keep = True
            for r in results:
                if abs(mx - r[0]) < min_dist and abs(my - r[1]) < min_dist:
                    keep = False
                    break
            if keep:
                results.append(m)
        return results

    def find_board(self):
        screen = self.capture.get_screenshot()
        debug_img = screen.copy()

        # 1. Load anchor templates (freshly captured, match current cell size)
        tl_tmpl = self.matcher.templates.get("board_tl.png")
        br_tmpl = self.matcher.templates.get("board_br.png")
        if tl_tmpl is None or br_tmpl is None:
            logging.error("Anchor templates board_tl.png / board_br.png not loaded")
            cv2.imwrite("debug_no_anchor_templates.png", debug_img)
            return None

        tmpl_h, tmpl_w = tl_tmpl.shape[:2]

        # 2. Match both templates, take per-pixel max
        score_tl = cv2.matchTemplate(screen, tl_tmpl, cv2.TM_CCOEFF_NORMED)
        score_br = cv2.matchTemplate(screen, br_tmpl, cv2.TM_CCOEFF_NORMED)
        combined = np.maximum(score_tl, score_br)

        # 3. Find local maxima in combined score map
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(combined, kernel)

        threshold = 0.70
        peaks = (combined >= threshold) & (combined == dilated)
        y_peaks, x_peaks = np.where(peaks)
        raw_peaks = list(zip(x_peaks, y_peaks))

        if len(raw_peaks) < 10:
            threshold = 0.60
            peaks = (combined >= threshold) & (combined == dilated)
            y_peaks, x_peaks = np.where(peaks)
            raw_peaks = list(zip(x_peaks, y_peaks))

        if len(raw_peaks) < 10:
            logging.error(f"Only {len(raw_peaks)} cell peaks found (threshold={threshold})")
            cv2.imwrite("debug_scoremap.png", (combined * 255).astype(np.uint8))
            cv2.imwrite("debug_few_peaks.png", debug_img)
            return None

        # 4. NMS to deduplicate nearby peaks from the same cell
        filtered = self._nms(raw_peaks, max(tmpl_w, tmpl_h) * 0.4)

        # Convert template top-left → cell center
        centers = [(x + tmpl_w // 2, y + tmpl_h // 2) for x, y in filtered]

        # 5. Grid fitting
        centers_by_y = sorted(centers, key=lambda p: p[1])

        # Estimate vertical step from median Y-difference
        y_diffs = []
        for i in range(1, len(centers_by_y)):
            d = centers_by_y[i][1] - centers_by_y[i-1][1]
            if d > 5:
                y_diffs.append(d)
        step_y_est = float(np.median(y_diffs)) if y_diffs else tmpl_h

        # Group into rows
        y_tol = step_y_est * 0.45
        rows_list = []
        cur = [centers_by_y[0]]
        for pt in centers_by_y[1:]:
            if abs(pt[1] - cur[0][1]) <= y_tol:
                cur.append(pt)
            else:
                rows_list.append(sorted(cur, key=lambda p: p[0]))
                cur = [pt]
        rows_list.append(sorted(cur, key=lambda p: p[0]))

        # Keep rows that have close to expected column count
        valid_rows = [r for r in rows_list if len(r) >= self.expected_cols * 0.6]
        if len(valid_rows) < 2:
            logging.error(f"Could not form valid grid: {len(valid_rows)} valid rows out of {len(rows_list)}")
            cv2.imwrite("debug_scoremap.png", (combined * 255).astype(np.uint8))
            cv2.imwrite("debug_no_grid.png", debug_img)
            return None

        # Compute step from each valid row's X-span
        row_steps = []
        for r in valid_rows:
            row_steps.append((r[-1][0] - r[0][0]) / (self.expected_cols - 1))
        step = float(np.median(row_steps)) if row_steps else step_y_est

        # Origin and last-cell from first/last valid row
        cx0, cy0 = valid_rows[0][0]
        cx1, cy1 = valid_rows[-1][-1]

        rows = int(round((cy1 - cy0) / step)) + 1
        if rows < 2 or rows > self.expected_cols:
            logging.error(f"Implausible rows: {rows}")
            cv2.imwrite("debug_bad_rows.png", debug_img)
            return None

        # 6. Draw debug
        for r in valid_rows:
            cv2.circle(debug_img, r[0], 3, (0, 255, 0), -1)
            cv2.circle(debug_img, r[-1], 3, (0, 255, 0), -1)
        cv2.circle(debug_img, (cx0, cy0), 5, (0, 0, 255), -1)
        cv2.circle(debug_img, (cx1, cy1), 5, (0, 0, 255), -1)
        cv2.rectangle(debug_img,
                      (int(cx0 - step / 2), int(cy0 - step / 2)),
                      (int(cx1 + step / 2), int(cy1 + step / 2)),
                      (255, 0, 255), 1)
        cv2.imwrite("debug_calibration.png", debug_img)
        cv2.imwrite("debug_scoremap.png", (combined * 255).astype(np.uint8))

        logging.info(f"Found {len(centers)} cells → {self.expected_cols}x{rows} grid, step={step:.1f}, "
                     f"anchor={tmpl_w}x{tmpl_h}")

        # 7. Resize all tile templates to match anchor dimensions
        self.matcher.resize_tile_templates(tmpl_w, tmpl_h)

        origin_x, origin_y = self.capture.to_screen(cx0, cy0)
        return {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "cell_w": step,
            "cell_h": step,
            "visual_cell_w": tmpl_w,
            "visual_cell_h": tmpl_h,
            "win_ox": cx0,
            "win_oy": cy0,
            "window_offset_x": self.capture.window_offset_x,
            "window_offset_y": self.capture.window_offset_y,
            "rows": rows,
            "cols": self.expected_cols,
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
        crop_w = int(round(board_info.get('visual_cell_w',
                    closed_tmpl.shape[1] if closed_tmpl is not None else board_info.get('cell_w', 31))))
        crop_h = int(round(board_info.get('visual_cell_h',
                    closed_tmpl.shape[0] if closed_tmpl is not None else board_info.get('cell_h', 31))))

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
        crop_w = int(round(board_info.get('visual_cell_w',
                    closed_tmpl.shape[1] if closed_tmpl is not None else board_info.get('cell_w', 31))))
        crop_h = int(round(board_info.get('visual_cell_h',
                    closed_tmpl.shape[0] if closed_tmpl is not None else board_info.get('cell_h', 31))))

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

