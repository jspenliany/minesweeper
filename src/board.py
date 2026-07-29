import cv2
import numpy as np
import logging

class Board:
    def __init__(self, capture, matcher, expected_cols=30):
        self.capture = capture
        self.matcher = matcher
        self.expected_cols = expected_cols
        self.marked_cells = set()

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

    def find_board(self):
        screen = self.capture.get_screenshot()
        debug_img = screen.copy()
        h_screen, w_screen = screen.shape[:2]

        # 1. Edge detection → board rectangle
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
        cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(debug_img, "BOARD RECT", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        known_rows = {9: 9, 16: 16, 30: 16}
        expected_rows = known_rows.get(self.expected_cols, 16)

        # 2. Estimate step geometrically: board rect includes ~0.5 cell border on each side
        step_w = bw / (self.expected_cols + 0.5)
        step_h = bh / (expected_rows + 0.5)
        step = min(step_w, step_h)

        # 3. Center grid in board rect; verify all cells fit, shrink if needed
        for _ in range(5):
            margin_x = (bw - step * (self.expected_cols - 1)) / 2
            margin_y = (bh - step * (expected_rows - 1)) / 2
            cx0 = bx + margin_x
            cy0 = by + margin_y
            last_center_x = cx0 + step * (self.expected_cols - 1)
            last_center_y = cy0 + step * (expected_rows - 1)
            vis_half = max(step * 0.45, 5)
            if (last_center_x + vis_half <= w_screen and
                last_center_y + vis_half <= h_screen and
                cx0 - vis_half >= bx and cy0 - vis_half >= by):
                break
            step *= 0.97
        else:
            logging.error(f"Grid does not fit after 5 iterations (step={step:.1f})")
            return None

        rows = expected_rows
        cols = self.expected_cols
        cx1 = last_center_x
        cy1 = last_center_y

        visual_cell_w = max(8, int(round(step * 0.9)))
        visual_cell_h = max(8, int(round(step * 0.9)))

        logging.info(f"Board rect=({bx},{by},{bw},{bh}) → {cols}x{rows} grid, step={step:.1f}, "
                     f"cell_visual={visual_cell_w}x{visual_cell_h}")

        # 4. Draw debug
        cv2.circle(debug_img, (int(cx0), int(cy0)), 5, (0, 0, 255), -1)
        cv2.circle(debug_img, (int(cx1), int(cy1)), 5, (0, 0, 255), -1)
        cv2.rectangle(debug_img,
                      (int(cx0 - step / 2), int(cy0 - step / 2)),
                      (int(cx1 + step / 2), int(cy1 + step / 2)),
                      (255, 0, 255), 1)
        cv2.imwrite("debug_calibration.png", debug_img)

        # 5. Resize all tile templates
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
        }

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

                # 1. Closed-tile check via variance (robust to resize, no templates needed)
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                current_var = float(gray.var())
                if closed_baseline is not None:
                    ref = closed_baseline[r, c]
                    # Closed if current variance within 35% of baseline
                    closed = (abs(current_var - ref) / max(ref, 1.0)) < 0.35
                else:
                    # Fallback: closed tiles typically have var > 200 at this zoom
                    closed = current_var > 200
                if closed:
                    matrix[r, c] = -1
                    if closed_baseline is not None:
                        err = abs(current_var - ref) / max(ref, 1.0)
                        scores[r, c] = 1.0 - min(1.0, err)
                    else:
                        scores[r, c] = min(1.0, current_var / 5000.0)
                    continue

                # 3. Open cell — classify by color (robust to resize)
                num = self._classify_cell_by_color(cell_img)
                matrix[r, c] = num
                scores[r, c] = 0.0 if num == -2 else 1.0

        return matrix, scores

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
        avg = center.mean(axis=(0, 1))
        b, g, r = float(avg[0]), float(avg[1]), float(avg[2])

        # Blank (uniform light gray)
        if b > 200 and g > 200 and r > 200:
            return -2

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
