# Minesweeper Bot

Automates Chinese Minesweeper (30×16, 99 mines) via screen capture, template matching, constraint solving, and mouse input.

## Language

**Capture** (`src/capture.py`):
Takes screenshots of the Minesweeper window (client area via Win32 API) or full screen as fallback. Returns raw pixel data.
_Avoid_: Screenshot, Vision (when referring to capture specifically)

**Matcher** (`src/matcher.py`):
Matches cell templates against captured images using OpenCV `matchTemplate` to determine each cell's state (closed, opened number, mine, flag).
_Avoid_: Vision (when referring to matching)

**Board** (`src/board.py`):
Represents the game grid geometry (origin coordinates, cell step, rows, cols). Provides board-level `find_board()` (locate origin + measure spacing) and `analyze_board()` (classify each cell via Matcher). Tracks `marked_cells` to prevent redundant flagging. Depends on Capture and Matcher.
_Avoid_: Grid layout, vision pipeline

**Calibration**:
The combined process of (1) locating the board via anchor + tile matching to compute origin and cell spacing, and (2) verifying by clicking the four corners and checking they all opened. Spans `find_board` + `verify_calibration`.
_Avoid_: Setup, alignment test

**Controller**:
Translates logical cell coordinates (row, col) into screen pixel coordinates and executes mouse clicks via pyautogui. Does not make gameplay decisions.
_Avoid_: Clicker, input handler

**GameState**:
The bot's lifecycle, modeled as a state machine:
- `IDLE` — polling for Minesweeper window
- `START_GAME` — press F2 to start trial game
- `TEST_CALIBRATION` — locate board + click 4 corners + verify → F2 → FIRST_MOVE
- `FIRST_MOVE` — locate fresh board + click center → PLAYING
- `PLAYING` — analyze → solve → act loop, with dialog interruption
- `RESULT` — game over, handle dialog or find restart button
_Avoid_: Status, mode

**Solver** (`src/solver.py`):
Constraint-based perfect solver. Maintains an internal grid, applies deterministic rules (Case A / Case B), falls back to random guess when stuck. Returns `(action, coords, reason)`. Shares `marked_cells` set with Board to prevent re-flagging.
_Avoid_: AI, engine

**Reasoner** (`src/solver.py`):
Formats Solver's output into human-readable reasoning strings (`Reasoner.format()`). Separate from Solver to keep decision logic free of presentation concerns.
_Avoid_: Logger, output formatter
