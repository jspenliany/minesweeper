import numpy as np
import logging
from src.timer import timer

class Solver:
    def __init__(self, rows, cols, marked_cells=None):
        self.rows = rows
        self.cols = cols
        # Status: -1: Unknown, 0-8: Number, 9: Mine, 10: Flag
        self.grid = np.full((rows, cols), -1, dtype=int)
        # Share marked_cells set with Board to prevent re-marking
        self.marked_cells = marked_cells if marked_cells is not None else set()
        
    def update_grid(self, new_grid):
        """Update the internal state from the board analysis"""
        self.grid = new_grid.copy()
        for r, c in self.marked_cells:
            if self.grid[r, c] != 10 and self.grid[r, c] != 9:
                self.grid[r, c] = 10

    def get_neighbors(self, r, c):
        """Get coordinates of all 8 neighbors"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    @timer
    def solve(self):
        """
        Perfect solver logic.
        Returns: list of (action, coords, reasoning) tuples.
        Action: 'CLICK' (Safe), 'MARK' (Mine), 'GUESS' (Random), 'NONE'
        Batching: all deterministic CLICK actions from the same round are
        returned together so the caller can execute them in batch.
        """
        results = []

        # === Phase 1: T1 - Basic Logic ===
        # Priority: MARK before CLICK (mine flagging is never skipped)
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid[r, c]
                if val < 0 or val > 8: continue

                neighbors = self.get_neighbors(r, c)
                unknowns = [n for n in neighbors if self.grid[n[0], n[1]] == -1]
                mines = [n for n in neighbors if self.grid[n[0], n[1]] in (9, 10)]
                open_unrec = [n for n in neighbors if self.grid[n[0], n[1]] == -3]

                if not unknowns and not open_unrec:
                    continue

                # T1-A: unknowns == remaining mines → all must be mines
                if len(unknowns) == (val - len(mines)):
                    unmarked = [u for u in unknowns if u not in self.marked_cells]
                    if not unmarked:
                        continue
                    first_u = unmarked[0]
                    return [('MARK', first_u,
                             f"Cell ({r},{c}) has value {val}, and exactly {len(unknowns)} unknown neighbors left. All must be mines.")]

                # T1-B: mines == val → all remaining unknowns are safe
                if len(mines) == val:
                    for u in unknowns:
                        results.append(('CLICK', u,
                            f"Cell ({r},{c}) has value {val}, and {len(mines)} mines already found around it. All other neighbors are safe."))

        if results:
            # Deduplicate CLICKs (same cell may appear from multiple number cells)
            seen = set()
            unique = []
            for a, c, r in results:
                if a == 'CLICK' and c in seen:
                    continue
                if a == 'CLICK':
                    seen.add(c)
                unique.append((a, c, r))
            return unique

        # === Phase 2: T2 - Pairwise Constraint Overlap ===
        cell_list = [(r, c) for r in range(self.rows) for c in range(self.cols)
                     if 0 <= self.grid[r, c] <= 8]

        for i in range(len(cell_list)):
            r1, c1 = cell_list[i]
            v1 = self.grid[r1, c1]
            n1 = self.get_neighbors(r1, c1)
            u1 = {n for n in n1 if self.grid[n] == -1}
            m1 = sum(1 for n in n1 if self.grid[n] in (9, 10))
            r1_rem = v1 - m1
            if r1_rem <= 0 or not u1:
                continue

            for j in range(i + 1, len(cell_list)):
                r2, c2 = cell_list[j]
                v2 = self.grid[r2, c2]
                n2 = self.get_neighbors(r2, c2)
                u2 = {n for n in n2 if self.grid[n] == -1}
                m2 = sum(1 for n in n2 if self.grid[n] in (9, 10))
                r2_rem = v2 - m2
                if r2_rem <= 0 or not u2:
                    continue

                x_only = u1 - u2
                y_only = u2 - u1
                common = u1 & u2
                x, y, c = len(x_only), len(y_only), len(common)
                if x + y + c == 0:
                    continue

                k_min = max(0, r1_rem - x, r2_rem - y)
                k_max = min(r1_rem, r2_rem, c)
                if k_min != k_max:
                    continue
                k = k_min

                # X_only mines
                if r1_rem - k == x and x > 0:
                    unmarked = [cell for cell in x_only if cell not in self.marked_cells]
                    if unmarked:
                        return [('MARK', unmarked[0],
                            f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: all {len(x_only)} cells unique to ({r1},{c1}) must be mines.")]

                # X_only safe
                if r1_rem - k == 0 and x > 0:
                    for cell in x_only:
                        results.append(('CLICK', cell,
                            f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: all {len(x_only)} cells unique to ({r1},{c1}) are safe."))

                # Y_only mines
                if r2_rem - k == y and y > 0:
                    unmarked = [cell for cell in y_only if cell not in self.marked_cells]
                    if unmarked:
                        return [('MARK', unmarked[0],
                            f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: all {len(y_only)} cells unique to ({r2},{c2}) must be mines.")]

                # Y_only safe
                if r2_rem - k == 0 and y > 0:
                    for cell in y_only:
                        results.append(('CLICK', cell,
                            f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: all {len(y_only)} cells unique to ({r2},{c2}) are safe."))

                # Common safe
                if k == 0 and c > 0:
                    for cell in common:
                        results.append(('CLICK', cell,
                            f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: no mines in shared area, all {c} are safe."))

                # Common mines
                if k == c and c > 0:
                    unmarked = [cell for cell in common if cell not in self.marked_cells]
                    if unmarked:
                        return [('MARK', unmarked[0],
                            f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: all {c} shared cells must be mines.")]

        if results:
            seen = set()
            unique = []
            for a, c, r in results:
                if a == 'CLICK' and c in seen:
                    continue
                if a == 'CLICK':
                    seen.add(c)
                unique.append((a, c, r))
            return unique

        # === Phase 3: GUESS ===
        unknowns = [(r, c) for r in range(self.rows) for c in range(self.cols) if self.grid[r, c] == -1]
        if not unknowns:
            return [('NONE', None, "Board fully solved!")]
        import random
        guess_pos = random.choice(unknowns)
        return [('GUESS', guess_pos, f"No deterministic logic available. Choosing a random safe candidate at {guess_pos}.")]



class Reasoner:
    """Formats Solver output into human-readable reasoning strings."""

    @staticmethod
    def format(action, coords, reasoning):
        if action == 'NONE':
            return reasoning
        r, c = coords
        action_map = {'CLICK': 'Left-Click (Safe)', 'MARK': 'Right-Click (Mine)', 'GUESS': 'Guessing'}
        return f"[{action_map[action]}] at ({r}, {c}): {reasoning}"
