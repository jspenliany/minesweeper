"""Test digit-only matching on flag0055.png row 0."""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, "D:/workspace/minesweeper-bot")
from src.matcher import Matcher

SCREENSHOT = "D:/workspace/minesweeper-bot/flag0055.png"
TILES_DIR = "D:/workspace/minesweeper-bot/assets/tiles"

matcher = Matcher()
screen = cv2.imread(SCREENSHOT)
print(f"Screenshot: {screen.shape[1]}x{screen.shape[0]}")

# Find board anchors
tl_tmpl = matcher.templates.get("board_tl.png")
br_tmpl = matcher.templates.get("board_br.png")

def find(tmpl):
    res = cv2.matchTemplate(screen, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.8:
        return None
    h, w = tmpl.shape[:2]
    return max_loc[0], max_loc[1], w, h

tl = find(tl_tmpl)
br = find(br_tmpl)
if not tl or not br:
    print("ERROR: anchors not found")
    exit(1)

ax, ay, aw, ah = tl
bx, by, bw, bh = br
cx0, cy0 = ax + aw // 2, ay + ah // 2
cx1, cy1 = bx + bw // 2, by + bh // 2

step = 31.0
cols = int(round((cx1 - cx0) / step)) + 1
rows = int(round((cy1 - cy0) / step)) + 1
print(f"Board: {cols}x{rows}, TL=({cx0},{cy0}) BR=({cx1},{cy1})")

# Templates
closed_tmpl = matcher.templates.get("closed_tile.png")
open_tmpl = matcher.templates.get("open_blank.png")
crop_w = closed_tmpl.shape[1] if closed_tmpl is not None else int(round(step))
crop_h = closed_tmpl.shape[0] if closed_tmpl is not None else int(round(step))

ox, oy = cx0, cy0
MARGIN = 1
tile_th = 0.70
open_th = 0.35
mine_th = 0.50

VALS = {-1: "CLOSED", -2: "BLANK", -3: "OPEN_?", 0: "0", 1: "1", 2: "2",
        3: "3", 4: "4", 5: "5", 9: "MINE", 10: "FLAG"}

NUMBER_NAMES = ["1.png", "2.png", "3.png", "4.png", "5.png"]

def match_with_margin(cell_img, template, margin):
    if margin > 0 and template.shape[0] > 2*margin and template.shape[1] > 2*margin:
        inner = cell_img[margin:-margin, margin:-margin]
        tmpl_inner = template[margin:-margin, margin:-margin]
        res = cv2.matchTemplate(inner, tmpl_inner, cv2.TM_CCOEFF_NORMED)
    else:
        res = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val

print(f"\n{'Col':>3} {'Final':>8} {'BestSc':>7} {'1':>7} {'2':>7} {'3':>7} {'4':>7} {'5':>7} {'open':>7} {'mine':>7} {'flag':>7} {'closed':>7}")
print("-" * 90)

r = 7
for c in range(cols):
    x = int(ox - crop_w / 2 + c * step + 0.5)
    y = int(oy - crop_h / 2 + r * step + 0.5)

    if x < 0 or y < 0 or x + crop_w > screen.shape[1] or y + crop_h > screen.shape[0]:
        print(f"{c:>3} {'OUT':>8}")
        continue

    cell_img = screen[y: y + crop_h, x: x + crop_w]
    inner = cell_img[MARGIN:-MARGIN, MARGIN:-MARGIN]

    best_val = None
    best_score = -1.0

    # Open blank
    open_score = match_with_margin(cell_img, open_tmpl, MARGIN) if open_tmpl is not None else 0.0
    if open_score > open_th and open_score > best_score:
        best_score = open_score
        best_val = -2

    # Number (digit-only)
    num_scores = {}
    for name in NUMBER_NAMES:
        tmpl = matcher.templates.get(name.replace(".png", "_digit.png"))
        if tmpl is None:
            num_scores[name] = 0.0
            continue
        if inner.shape[0] < tmpl.shape[0] or inner.shape[1] < tmpl.shape[1]:
            num_scores[name] = 0.0
            continue
        res = cv2.matchTemplate(inner, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        num_scores[name] = score
        if score > tile_th and score > best_score:
            best_score = score
            best_val = matcher.map_value(name)

    # Mine/flag
    mine_score = match_with_margin(cell_img, matcher.templates.get("mine.png"), MARGIN) if matcher.templates.get("mine.png") is not None else 0.0
    flag_score = match_with_margin(cell_img, matcher.templates.get("flag.png"), MARGIN) if matcher.templates.get("flag.png") is not None else 0.0
    if mine_score > mine_th and mine_score > best_score:
        best_score = mine_score
        best_val = 9
    if flag_score > mine_th and flag_score > best_score:
        best_score = flag_score
        best_val = 10

    if best_val is not None:
        final = VALS.get(best_val, str(best_val))
    else:
        # Closed check
        closed_score = match_with_margin(cell_img, closed_tmpl, 0) if closed_tmpl is not None else 0.0
        closed_th = 0.70
        final = "CLOSED" if closed_score > closed_th else "OPEN_?"
        best_score = closed_score

    print(f"{c:>3} {final:>8} {best_score:>7.3f} "
          f"{num_scores.get('1.png',0):>7.3f} {num_scores.get('2.png',0):>7.3f} "
          f"{num_scores.get('3.png',0):>7.3f} {num_scores.get('4.png',0):>7.3f} "
          f"{num_scores.get('5.png',0):>7.3f} {open_score:>7.3f} "
          f"{mine_score:>7.3f} {flag_score:>7.3f} "
          f"{match_with_margin(cell_img, closed_tmpl, 0) if closed_tmpl is not None else 0.0:>7.3f}")
