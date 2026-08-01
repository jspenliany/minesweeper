import cv2
import numpy as np
import logging
from src.timer import timer

class Board:
    def __init__(self, capture, matcher, expected_cols=30):
        self.capture = capture
        self.matcher = matcher
        self.expected_cols = expected_cols
        self.marked_cells = set()
        self._cell_cache = {}

    def _find_board_contour(self, screen, debug_img):
        """Edge detection → find the board rectangle. Returns (bx, by, bw, bh) or None."""
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        known_rows = {9: 9, 16: 16, 30: 16}
        expected_rows = known_rows.get(self.expected_cols, 16)
        expected_aspect = self.expected_cols / expected_rows

        best_rect = None
        best_score = 0
        for method in range(3):
            if method == 0:
                edges = cv2.Canny(blurred, 30, 100)
            elif method == 1:
                edges = cv2.Canny(blurred, 10, 50)
            else:
                edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) != 4:
                    continue
                rx, ry, rw, rh = cv2.boundingRect(approx)
                if rw < 100 or rh < 100:
                    continue
                aspect = rw / rh
                aspect_score = max(0, 1.0 - abs(aspect - expected_aspect) / expected_aspect)
                area_score = (rw * rh) / (screen.shape[1] * screen.shape[0])
                score = aspect_score * 0.5 + area_score * 0.5
                if score > best_score:
                    best_score = score
                    best_rect = (rx, ry, rw, rh)

        return best_rect

    def _find_board_by_anchors(self, screen):
        """Use board_tl.png / board_br.png template matching to locate the grid.
        Returns (tl_x, tl_y, br_x, br_y, cell_w, cell_h, cols, rows, step) or None."""
        tl_tmpl = self.matcher.templates.get("board_tl.png")
        br_tmpl = self.matcher.templates.get("board_br.png")
        if tl_tmpl is None or br_tmpl is None:
            return None

        cell_w = tl_tmpl.shape[1]
        cell_h = tl_tmpl.shape[0]

        tl_matches = self.matcher.match_all(screen, "board_tl.png", threshold=0.7)
        br_matches = self.matcher.match_all(screen, "board_br.png", threshold=0.7)
        if not tl_matches or not br_matches:
            return None

        # TL corner: top-left-most match (smallest x+y)
        tl_matches.sort(key=lambda p: p[0] + p[1])
        tl_x, tl_y = tl_matches[0]

        # BR corner: bottom-right-most match (largest x+y)
        br_matches.sort(key=lambda p: -(p[0] + p[1]))
        br_x, br_y = br_matches[0]

        if br_x <= tl_x or br_y <= tl_y:
            return None

        dx = br_x - tl_x
        dy = br_y - tl_y

        # Detect grid size by trying common configurations
        best_info = None
        best_err = 1.0
        for n_cols, n_rows in [(30, 16), (16, 16), (9, 9)]:
            if n_cols < 2 or n_rows < 2:
                continue
            sx = dx / (n_cols - 1)
            sy = dy / (n_rows - 1)
            if max(sx, sy) == 0:
                continue
            ratio = min(sx, sy) / max(sx, sy)
            if ratio > 0.80:
                err = 1.0 - ratio
                if err < best_err:
                    best_err = err
                    step = (sx + sy) / 2.0
                    best_info = (tl_x, tl_y, br_x, br_y, cell_w, cell_h, n_cols, n_rows, step)

        if best_info is None:
            return None

        logging.info(f"Found board via anchor matching: TL=({tl_x},{tl_y}) BR=({br_x},{br_y}) "
                     f"{best_info[6]}x{best_info[7]} grid, step={best_info[8]:.1f}, cell={cell_w}x{cell_h}")
        return best_info

    @timer
    def find_board(self):
        screen = self.capture.get_screenshot()
        # A newly located board invalidates any cached per-cell classifications
        # from the previous game.
        self._cell_cache = {}
        debug_img = screen.copy()
        h_screen, w_screen = screen.shape[:2]

        # 1. Anchor template matching (primary)
        anchor_info = self._find_board_by_anchors(screen)

        if anchor_info is not None:
            tl_x, tl_y, br_x, br_y, cell_w, cell_h, cols, rows, step = anchor_info

            visual_cell_w = cell_w
            visual_cell_h = cell_h
            cx0 = tl_x + cell_w / 2.0
            cy0 = tl_y + cell_h / 2.0
            cx1 = br_x + cell_w / 2.0
            cy1 = br_y + cell_h / 2.0

            board_rect_label = f"anchors TL=({tl_x},{tl_y}) BR=({br_x},{br_y})"
        else:
            # 2. Fallback: edge detection → board rectangle
            board_rect = self._find_board_contour(screen, debug_img)
            if board_rect is None:
                # Fallback: color-based — Minesweeper board is light gray
                hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
                lower = np.array([0, 0, 180])
                upper = np.array([180, 30, 255])
                mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                known_rows = {9: 9, 16: 16, 30: 16}
                expected_rows = known_rows.get(self.expected_cols, 16)
                expected_aspect = self.expected_cols / expected_rows
                best_rect = None
                best_score = 0
                for cnt in cnts:
                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    if rw < 100 or rh < 100:
                        continue
                    aspect = rw / rh
                    aspect_score = max(0, 1.0 - abs(aspect - expected_aspect) / expected_aspect)
                    area_score = (rw * rh) / (w_screen * h_screen)
                    score = aspect_score * 0.5 + area_score * 0.5
                    if score > best_score:
                        best_score = score
                        best_rect = (rx, ry, rw, rh)
                if best_rect and best_score > 0.3:
                    board_rect = best_rect
                    logging.info(f"Found board via color mask: ({rx},{ry},{rw},{rh})")

            if board_rect is None:
                logging.warning("All detection methods failed; using full screenshot as board rect")
                board_rect = (0, 0, w_screen, h_screen)

            bx, by, bw, bh = board_rect
            board_rect_label = f"({bx},{by},{bw},{bh})"

            known_rows = {9: 9, 16: 16, 30: 16}
            expected_rows = known_rows.get(self.expected_cols, 16)

            # Estimate step geometrically
            step_w = bw / (self.expected_cols + 0.5)
            step_h = bh / (expected_rows + 0.5)
            step = min(step_w, step_h)

            for _ in range(5):
                margin_x = (bw - step * (self.expected_cols - 1)) / 2
                margin_y = (bh - step * (expected_rows - 1)) / 2
                cx0 = bx + margin_x
                cy0 = by + margin_y
                cx1 = cx0 + step * (self.expected_cols - 1)
                cy1 = cy0 + step * (expected_rows - 1)
                vis_half = max(step * 0.45, 5)
                if (cx1 + vis_half <= w_screen and
                    cy1 + vis_half <= h_screen and
                    cx0 - vis_half >= bx and cy0 - vis_half >= by):
                    break
                step *= 0.97
            else:
                logging.error(f"Grid does not fit after 5 iterations (step={step:.1f})")
                return None

            rows = expected_rows
            cols = self.expected_cols

            # Use anchor template dimensions for cell_visual if available
            tl_tmpl = self.matcher.templates.get("board_tl.png")
            if tl_tmpl is not None:
                visual_cell_w = tl_tmpl.shape[1]
                visual_cell_h = tl_tmpl.shape[0]
            else:
                visual_cell_w = max(8, int(round(step * 0.9)))
                visual_cell_h = max(8, int(round(step * 0.9)))

        cv2.rectangle(debug_img, (int(cx0 - step / 2), int(cy0 - step / 2)),
                      (int(cx1 + step / 2), int(cy1 + step / 2)), (0, 255, 0), 2)
        cv2.putText(debug_img, f"BOARD {board_rect_label}", (int(cx0), int(cy0) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(debug_img, (int(cx0), int(cy0)), 5, (0, 0, 255), -1)
        cv2.circle(debug_img, (int(cx1), int(cy1)), 5, (0, 0, 255), -1)
        cv2.rectangle(debug_img,
                      (int(cx0 - step / 2), int(cy0 - step / 2)),
                      (int(cx1 + step / 2), int(cy1 + step / 2)),
                      (255, 0, 255), 1)
        cv2.imwrite("debug_calibration.png", debug_img)

        logging.info(f"Board {board_rect_label} → {cols}x{rows} grid, step={step:.1f}, "
                     f"cell_visual={visual_cell_w}x{visual_cell_h}")

        self.matcher.resize_tile_templates(visual_cell_w, visual_cell_h)

        origin_x, origin_y = self.capture.to_screen(int(cx0), int(cy0))
        return {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "cell_w": step,
            "cell_h": step,
            "visual_cell_w": visual_cell_w,
            "visual_cell_h": visual_cell_h,
            "win_ox": int(cx0),
            "win_oy": int(cy0),
            "window_offset_x": self.capture.window_offset_x,
            "window_offset_y": self.capture.window_offset_y,
            "rows": rows,
            "cols": cols,
            "anchor_matched": anchor_info is not None,
        }

    @timer
    def compute_closed_baseline(self, board_info):
        """Compute per-cell grayscale variance as a 'closed tile' baseline.
           Closed tiles have high variance (3D bevel: bright highlight + dark shadow)."""
        screen = self.capture.get_screenshot()
        rows = board_info.get('rows', 9)
        cols = board_info.get('cols', 9)
        crop_w = int(round(board_info.get('visual_cell_w', 32)))
        crop_h = int(round(board_info.get('visual_cell_h', 32)))

        ox = board_info['origin_x'] - board_info['window_offset_x']
        oy = board_info['origin_y'] - board_info['window_offset_y']

        baseline = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                x = int(ox - crop_w / 2 + c * board_info['cell_w'] + 0.5)
                y = int(oy - crop_h / 2 + r * board_info['cell_h'] + 0.5)
                if (x < 0 or y < 0 or
                    x + crop_w > screen.shape[1] or
                    y + crop_h > screen.shape[0]):
                    baseline[r, c] = 500.0
                    continue
                cell_img = screen[y : y + crop_h, x : x + crop_w]
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                baseline[r, c] = float(gray.var())
        return baseline

    @timer
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

        # Use stored calibration offset to recover cx0 (client-area coordinate),
        # which is invariant under window movement (screenshot is always client area).
        ox = board_info['origin_x'] - board_info['window_offset_x']
        oy = board_info['origin_y'] - board_info['window_offset_y']

        closed_baseline = board_info.get('closed_baseline')

        cache = self._cell_cache

        for r in range(rows):
            for c in range(cols):
                x = int(ox - crop_w / 2 + c * board_info['cell_w'] + 0.5)
                y = int(oy - crop_h / 2 + r * board_info['cell_h'] + 0.5)
            #    logging.debug(f"Cell ({r},{c}) at crop ({x},{y})")
                if (x < 0 or y < 0 or
                    x + crop_w > screen.shape[1] or
                    y + crop_h > screen.shape[0]):
                    logging.warning(f"Cell ({r},{c}) at crop ({x},{y}) outside screenshot bounds "
                                    f"({screen.shape[1]}x{screen.shape[0]}). Marking as closed.")
                    matrix[r, c] = -1
                    scores[r, c] = 0.0
                    continue

                cell_img = screen[y : y + crop_h, x : x + crop_w]

                # Incremental cache: most cells are pixel-identical between
                # consecutive analyses (only the cascade region changes after a
                # click). Reuse the previous classification when unchanged; the
                # pixel-exact compare keeps this safe — any change at all falls
                # through to full classification below.
                cached = cache.get((r, c))
                if cached is not None:
                    prev_img, prev_val, prev_score = cached
                    if prev_img.shape == cell_img.shape and np.array_equal(prev_img, cell_img):
                        matrix[r, c] = prev_val
                        scores[r, c] = prev_score
                        continue

                # 1. Flag detection (closed tile with a red flag icon)
                # Rule: closed tiles have high variance; a red flag in the center
                #       (high R, low G/B) distinguishes flag from red digit 3 on an open tile.
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                current_var = float(gray.var())
                closed = False
                if closed_baseline is not None:
                    ref = closed_baseline[r, c]
                    closed = (abs(current_var - ref) / max(ref, 1.0)) < 0.35
                else:
                    closed = current_var > 200

                if closed:
                    matrix[r, c] = -1
                    if closed_baseline is not None:
                        err = abs(current_var - ref) / max(ref, 1.0)
                        scores[r, c] = 1.0 - min(1.0, err)
                    else:
                        scores[r, c] = min(1.0, current_var / 5000.0)
                    cache[(r, c)] = (cell_img.copy(), matrix[r, c], scores[r, c])
                    continue

                # 3. Open cell — try digit template matching first
                num = self._classify_cell_by_digit_templates(cell_img)
                if num is None:
                    # Fallback: color-based blank / unknown
                    num = self._classify_cell_by_color(cell_img)
                matrix[r, c] = num
                scores[r, c] = 0.0 if num == -2 else 1.0 if num >= 0 else 0.5
                cache[(r, c)] = (cell_img.copy(), num, scores[r, c])

        return matrix, scores

    @timer
    def _classify_cell_by_digit_templates(self, cell_img):
        """Match inner region (MARGIN=1) against cropped digit templates.
        MARGIN=1 removes the outermost pixel (cell border/divider) while
        keeping enough sliding room for large digit templates like 5 (25×30)."""
        MARGIN = 1
        if cell_img.shape[0] <= 2 * MARGIN or cell_img.shape[1] <= 2 * MARGIN:
            return None
        inner = cell_img[MARGIN:-MARGIN, MARGIN:-MARGIN]

        # Blank guard: TM_CCOEFF_NORMED responds mainly to shape/intensity
        # patterns and discriminates color only weakly, so a blank cell's gray
        # bevel can spuriously correlate with a digit shape. A real digit always
        # has hundreds of strongly-colored pixels (saturation > 40); blank
        # bevels are gray (saturation ~ 0). Bail out before matching.
        arr = inner.astype(np.int32)
        sat = arr.max(axis=2) - arr.min(axis=2)
        if int((sat > 40).sum()) < 50:
            return None

        # Acceptance threshold 0.60, not 0.40: on flat low-texture cells the
        # normalized correlation can spike to ~0.40-0.41 for a wrong digit (a
        # blank tile spuriously matching e.g. "4"), while a real digit scores
        # ~0.90+ on its own template. 0.60 cleanly separates the two.
        best_num = None
        best_score = 0.60
        for name in ("1.png", "2.png", "3.png", "4.png", "5.png", "6.png"):
            tmpl = self.matcher.templates.get(name.replace(".png", "_digit.png"))
            if tmpl is None:
                continue
            if inner.shape[0] < tmpl.shape[0] or inner.shape[1] < tmpl.shape[1]:
                continue
            res = cv2.matchTemplate(inner, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)
            if score > best_score:
                best_score = score
                best_num = self.matcher.map_value(name)
        return best_num

    @staticmethod
    def _classify_cell_by_color(cell_img):
        """Classify an open cell by the dominant color in its center region.
        Returns: -2 (blank), 1-8 (number), -3 (unrecognized)."""
        h, w = cell_img.shape[:2]
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
        if y2 <= y1 or x2 <= x1:
            return -3
        center = cell_img[y1:y2, x1:x2]
        std = center.std(axis=(0, 1))
        b_std, g_std, r_std = float(std[0]), float(std[1]), float(std[2])

        # Blank (uniform light gray) — low std in all channels
        if b_std < 15 and g_std < 15 and r_std < 15:
            avg = center.mean(axis=(0, 1))
            b, g, r = float(avg[0]), float(avg[1]), float(avg[2])
            if b > 180 and g > 180 and r > 180:
                return -2
            return -3

        # There's a digit — find the pixel farthest from gray to get the digit color
        gray_avg = center.mean(axis=(0, 1))
        gray = np.full_like(center, gray_avg, dtype=center.dtype)
        diff = np.abs(center.astype(np.float32) - gray.astype(np.float32)).sum(axis=2)
        flat_idx = diff.argmax()
        max_y = flat_idx // center.shape[1]
        max_x = flat_idx % center.shape[1]
        b, g, r = float(center[max_y, max_x, 0]), float(center[max_y, max_x, 1]), float(center[max_y, max_x, 2])

        # Known digit BGR centroids
        colors = {
            1: (255, 0, 0),
            2: (0, 128, 0),
            3: (0, 0, 255),
            4: (128, 0, 0),
            5: (0, 0, 128),
            6: (0, 128, 128),
            7: (0, 0, 0),
            8: (128, 128, 128),
        }

        best_num = -3
        best_dist = float('inf')
        for num, (cb, cg, cr) in colors.items():
            dist = (b - cb) ** 2 + (g - cg) ** 2 + (r - cr) ** 2
            if dist < best_dist:
                best_dist = dist
                best_num = num

        if best_dist > 6000:
            return -3
        return best_num

    def mark_cell(self, r, c):
        self.marked_cells.add((r, c))
