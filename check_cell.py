import cv2
import numpy as np
import sys
sys.path.insert(0, "D:/workspace/minesweeper-bot")
from src.matcher import Matcher

screen = cv2.imread("D:/workspace/minesweeper-bot/flag0055.png")
matcher = Matcher()

tl = matcher.find_image(screen, "board_tl.png")
br = matcher.find_image(screen, "board_br.png")
ax, ay, aw, ah = tl
bx, by, bw, bh = br
cx0, cy0 = ax + aw // 2, ay + ah // 2
cx1, cy1 = bx + bw // 2, by + bh // 2

step = 31.0
ox, oy = cx0, cy0
closed_tmpl = matcher.templates.get("closed_tile.png")
crop_w = closed_tmpl.shape[1] if closed_tmpl is not None else int(round(step))
crop_h = closed_tmpl.shape[0] if closed_tmpl is not None else int(round(step))

def match_margin(cell_img, template, margin):
    if margin > 0 and template.shape[0] > 2*margin and template.shape[1] > 2*margin:
        inner = cell_img[margin:-margin, margin:-margin]
        t_inner = template[margin:-margin, margin:-margin]
        res = cv2.matchTemplate(inner, t_inner, cv2.TM_CCOEFF_NORMED)
    else:
        res = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
    return cv2.minMaxLoc(res)[1]

for r, c, label in [(3, 17, "(3,17)"), (2, 18, "(2,18)"), (2, 17, "(2,17)")]:
    x = int(ox - crop_w/2 + c*step + 0.5)
    y = int(oy - crop_h/2 + r*step + 0.5)
    cell = screen[y:y+crop_h, x:x+crop_w]
    inner = cell[1:-1, 1:-1]
    print(f"\n=== Cell {label} ===")
    print(f"  closed_tile: {match_margin(cell, closed_tmpl, 0):.4f}")
    print(f"  open_blank:  {match_margin(cell, matcher.templates.get('open_blank.png'), 1):.4f}")
    mine_tmpl = matcher.templates.get("mine.png")
    if mine_tmpl is not None:
        print(f"  mine:        {match_margin(cell, mine_tmpl, 1):.4f}")
    flag_tmpl = matcher.templates.get("flag.png")
    if flag_tmpl is not None:
        print(f"  flag:        {match_margin(cell, flag_tmpl, 1):.4f}")
    for name in ["1.png","2.png","3.png","4.png","5.png"]:
        tmpl = matcher.templates.get(name.replace(".png", "_digit.png"))
        if tmpl is None:
            continue
        res = cv2.matchTemplate(inner, tmpl, cv2.TM_CCOEFF_NORMED)
        score = cv2.minMaxLoc(res)[1]
        print(f"  {name[:-4]}:          {score:.4f}")
