import numpy as np
import logging

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

    def solve(self):
        """
        Perfect solver logic.
        Returns: (action, coords, reasoning)
        Action: 'CLICK' (Safe), 'MARK' (Mine), 'GUESS' (Random), 'NONE'
        """
        # 1. Basic Logic: T1 (Simple count matching)
        # Priority: always return MARK before CLICK, so mine flagging
        # is never skipped by an earlier cell's "safe click" deduction.
        
        deferred_clicks = []
        
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid[r, c]
                if val < 0 or val > 8: continue # Only process number cells
                
                neighbors = self.get_neighbors(r, c)
                unknowns = [n for n in neighbors if self.grid[n[0], n[1]] == -1]
                mines = [n for n in neighbors if self.grid[n[0], n[1]] == 9 or self.grid[n[0], n[1]] == 10]
                open_unrec = [n for n in neighbors if self.grid[n[0], n[1]] == -3]
                
                # -3 is open but unrecognized (safe, not a mine)
                if not unknowns and not open_unrec:
                    continue
                
                # Case A: All remaining unknowns must be mines
                if len(unknowns) == (val - len(mines)):
                    # Skip cells we've already tried marking (avoids flag→? cycling)
                    unmarked = [u for u in unknowns if u not in self.marked_cells]
                    if not unmarked:
                        continue
                    first_u = unmarked[0]
                    reason = f"Cell ({r},{c}) has value {val}, and exactly {len(unknowns)} unknown neighbors left. All must be mines."
                    return 'MARK', first_u, reason
                
                # Case B: All remaining unknowns must be safe
                if len(mines) == val:
                    deferred_clicks.append((r, c, val, len(mines), unknowns[0]))
        
        # Return first deferred CLICK (if any) after exhausting all MARK opportunities
        if deferred_clicks:
            r, c, val, mines_found, first_u = deferred_clicks[0]
            reason = f"Cell ({r},{c}) has value {val}, and {mines_found} mines already found around it. All other neighbors are safe."
            return 'CLICK', first_u, reason

        # 2. Advanced Logic: T2 (Pairwise constraint overlap)
        # For any two number cells (A,B), we know:
        #   R_A = remaining mines in U_A = val_A - mines_found_A
        #   R_B = remaining mines in U_B = val_B - mines_found_B
        #   X_only = U_A \ U_B,  Y_only = U_B \ U_A,  common = U_A ∩ U_B
        # Let k = mines in common → k_min ≤ k ≤ k_max
        #   k_min = max(0, R_A - |X_only|, R_B - |Y_only|)
        #   k_max = min(R_A, R_B, |common|)
        # If k_min == k_max, k is forced → deduce mines/safe in X_only, Y_only, common

        cell_list = [(r, c) for r in range(self.rows) for c in range(self.cols)
                     if 0 <= self.grid[r, c] <= 8]
        deferred_clicks = []

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

                # X_only: r1_rem - k mines
                if r1_rem - k == x and x > 0:
                    unmarked = [cell for cell in x_only if cell not in self.marked_cells]
                    if unmarked:
                        return 'MARK', unmarked[0], \
                            f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: all {len(x_only)} cells unique to ({r1},{c1}) must be mines."

                if r1_rem - k == 0 and x > 0:
                    deferred_clicks.append((list(x_only)[0],
                        f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: all {len(x_only)} cells unique to ({r1},{c1}) are safe."))

                # Y_only: r2_rem - k mines
                if r2_rem - k == y and y > 0:
                    unmarked = [cell for cell in y_only if cell not in self.marked_cells]
                    if unmarked:
                        return 'MARK', unmarked[0], \
                            f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: all {len(y_only)} cells unique to ({r2},{c2}) must be mines."

                if r2_rem - k == 0 and y > 0:
                    deferred_clicks.append((list(y_only)[0],
                        f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: all {len(y_only)} cells unique to ({r2},{c2}) are safe."))

                # Common: k mines
                if k == 0 and c > 0:
                    deferred_clicks.append((list(common)[0],
                        f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: no mines in shared area, all {c} are safe."))

                if k == c and c > 0:
                    unmarked = [cell for cell in common if cell not in self.marked_cells]
                    if unmarked:
                        return 'MARK', unmarked[0], \
                            f"Constraint ({r1},{c1})={v1} & ({r2},{c2})={v2}: all {c} shared cells must be mines."

        if deferred_clicks:
            cell, reason = deferred_clicks[0]
            return 'CLICK', cell, reason

        # 3. GUESS: Last resort if no deterministic action found
        unknowns = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == -1:
                    unknowns.append((r, c))
        
        if not unknowns:
            return 'NONE', None, "Board fully solved!"
            
        import random
        guess_pos = random.choice(unknowns)
        return 'GUESS', guess_pos, f"No deterministic logic available. Choosing a random safe candidate at {guess_pos}."



class Reasoner:
    """Formats Solver output into human-readable reasoning strings."""

    @staticmethod
    def format(action, coords, reasoning):
        if action == 'NONE':
            return reasoning
        r, c = coords
        action_map = {'CLICK': 'Left-Click (Safe)', 'MARK': 'Right-Click (Mine)', 'GUESS': 'Guessing'}
        return f"[{action_map[action]}] at ({r}, {c}): {reasoning}"
