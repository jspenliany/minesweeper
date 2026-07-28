import cv2
import numpy as np
import logging

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
        # Template is 28x28 (inner cell), full cell is 30x30, extend 1px per side
        cv2.rectangle(debug_img, (ax - 1, ay - 1), (ax + aw + 1, ay + ah + 1), (0, 255, 0), 2)
        cv2.putText(debug_img, "Anchor=Cell(0,0)", (ax - 1, ay - 1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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

        tol = 4
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
        n_cells = len(row_xs)
        logging.info(f"Row0 NMS: {n_cells} cells, xs={row_xs[:3]}...{row_xs[-3:]}")

        # Apply user's formula:
        #   Cell side = x px, gap = 1 px → step = x + 1
        #   Total width = 30*x + 31, total height = 16*x + 17
        # Derive x from NMS span = step * (n_cells - 1)
        if n_cells >= 2:
            span = row_xs[-1] - row_xs[0]
            step_approx = span / (n_cells - 1)
            x = int(round(step_approx - 1))          # cell side (integer)
            cell_step_x = float(x + 1)                # step = x + 1
            cell_step_y = float(x + 1)
        else:
            x = tw                                  # fallback to template size
            cell_step_x = float(x + 1)
            cell_step_y = float(x + 1)

        total_w = self.expected_cols * x + (self.expected_cols + 1) * 1  # 30x + 31
        total_h = 16 * x + (16 + 1) * 1                                    # 16x + 17
        board_l = ax - (x - tw) // 2     # full cell(0,0) top-left
        board_t = ay - (x - th) // 2
        logging.info(f"Derived x={x}, step={cell_step_x:.1f}, board ({total_w}×{total_h}) "
                     f"at window ({board_l}, {board_t})")
        logging.info(f"Row0 NMS: {n_cells} cells, span={span}, step_approx={step_approx:.3f}")

        board_right = board_l + total_w
        board_bottom = board_t + total_h
        if board_right > screen.shape[1] or board_bottom > screen.shape[0]:
            logging.warning(f"Board extends beyond screenshot: right={board_right}, bottom={board_bottom}, "
                           f"size=({screen.shape[1]}x{screen.shape[0]})")

        # Full cell(0,0) center = inner-region center = anchor center
        origin_x, origin_y = self.capture.to_screen(ax + tw // 2, ay + th // 2)
        win_ox = ax + tw // 2   # window-relative origin (no conversion needed)
        win_oy = ay + th // 2

        # Debug marker at full cell(0,0) center
        cx = ax + tw // 2
        cy = ay + th // 2
        cv2.circle(debug_img, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(debug_img, f"Cell(0,0) x={x}", (cx + int(round(cell_step_x)) + 5, cy + int(round(cell_step_y // 2))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for mx, my in matches:
            cv2.rectangle(debug_img, (mx - 1, my - 1), (mx + tw + 1, my + th + 1), (255, 0, 255), 1)

        cv2.imwrite("debug_calibration.png", debug_img)
        logging.info(f"Screen origin ({origin_x}, {origin_y})")

        return {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "cell_w": cell_step_x,
            "cell_h": cell_step_y,
            "win_ox": win_ox,
            "win_oy": win_oy,
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
        crop_w = tile_template.shape[1] if tile_template is not None else int(round(board_info['cell_w']))
        crop_h = tile_template.shape[0] if tile_template is not None else int(round(board_info['cell_h']))

        matrix = np.full((rows, cols), -1, dtype=int)
        scores = np.full((rows, cols), 0.0, dtype=float)

        ox = board_info.get('win_ox', board_info['origin_x'] - board_info.get('window_offset_x', 0))
        oy = board_info.get('win_oy', board_info['origin_y'] - board_info.get('window_offset_y', 0))

        for r in range(rows):
            for c in range(cols):
                col_xs = board_info.get('col_xs')
                row_ys = board_info.get('row_ys')
                if col_xs and c < len(col_xs):
                    x = col_xs[c]
                else:
                    x = int(round(ox - crop_w / 2 + c * board_info['cell_w']))
                if row_ys and r < len(row_ys):
                    y = row_ys[r]
                else:
                    y = int(round(oy - crop_h / 2 + r * board_info['cell_h']))

                if (x < 0 or y < 0 or
                    x + crop_w > screen.shape[1] or
                    y + crop_h > screen.shape[0]):
                    logging.warning(f"Cell ({r},{c}) at crop ({x},{y}) outside screenshot bounds ({screen.shape[1]}x{screen.shape[0]}). Marking as closed.")
                    matrix[r, c] = -1
                    scores[r, c] = 0.0
                    continue

                cell_img = screen[y : y + crop_h, x : x + crop_w]

                matched = False
                for name, template in self.matcher.templates.items():
                    logging.info(f"Cell ({r},{c}) Matching {name}...before filtering. close , blank")
                    if not name.endswith('.png') or name in ("closed_tile.png", "open_blank.png", "blank.png"):
                        continue
                    logging.info(f"Matching {name}...before filtering. shape {cell_img.shape}")
                    if cell_img.shape[0] < template.shape[0] or cell_img.shape[1] < template.shape[1]:
                        continue
                    score = self.matcher.match_cell(cell_img, name)
                    logging.info(f"Matching {name}...after score {score}")
                    if score > self.cell_threshold:
                        val = self.matcher.map_value(name)
                        matrix[r, c] = val
                        scores[r, c] = score
                        matched = True
                        break

                if matched:
                    continue

                # Closed-tile template matching (works with exact NMS positions)
                closed_score = self.matcher.match_cell(cell_img, "closed_tile.png")
                open_score = self.matcher.match_cell(cell_img, "open_blank.png")
                if closed_score > 0.70:
                    matrix[r, c] = -1
                    scores[r, c] = closed_score
                elif open_score > 0.95:
                    matrix[r, c] = -2
                    scores[r, c] = open_score
                else:
                    # Fallback: pixel variance
                    _, stddev = cv2.meanStdDev(cell_img)
                    variance = np.mean(stddev)
                    if variance > 7.0:
                        matrix[r, c] = -1
                        scores[r, c] = closed_score
                    else:
                        matrix[r, c] = -2
                        scores[r, c] = open_score

        return matrix, scores

    def mark_cell(self, r, c):
        self.marked_cells.add((r, c))

    def apply_marks(self, grid):
        grid = grid.copy()
        for r, c in self.marked_cells:
            if grid[r, c] != 10 and grid[r, c] != 9:
                grid[r, c] = 10
        return grid
