# Calibration: test on trial game, then F2 for fresh game

Calibration first runs on a trial game (started by F2) to click the four corners and verify the origin and cell spacing are correct. Only after verification passes does the bot press F2 again to start a fresh, untouched game for actual play.

**Why not calibrate once and play directly?** The four-corner click changes the board state (opens cells), reducing template coverage for `analyze_board`. A fresh game guarantees all 480 cells are closed, giving `analyze_board` maximal data for classification.

**Why not verify with a static screenshot instead of clicking?** Clicking is the ground truth — it proves the coordinates are correct because the game responds. Static screenshot verification would require pre-labelled data and wouldn't catch offset errors that only manifest at runtime.
