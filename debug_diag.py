import sys, os, cv2, numpy as np, time, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, r'D:\workspace\minesweeper-bot')
from src.capture import Capture
from src.matcher import Matcher
from src.board import Board

c = Capture()
m = Matcher()
b = Board(c, m)

info = b.find_board()
if info is None:
    print('Board not found')
    sys.exit(1)

rows = info['rows']
cols = info['cols']
step = info['cell_w']
win_ox = info.get('win_ox', info['origin_x'] - c.window_offset_x)
win_oy = info.get('win_oy', info['origin_y'] - c.window_offset_y)

print(f"cols={cols} rows={rows}")
print(f"win_ox={win_ox} win_oy={win_oy}")
print(f"cell_w={step} cell_h={info['cell_h']}")

# analyze board
board_info = {
    'origin_x': info['origin_x'], 'origin_y': info['origin_y'],
    'cell_w': step, 'cell_h': info['cell_h'],
    'rows': rows, 'cols': cols,
    'win_ox': win_ox, 'win_oy': win_oy,
}
baseline = b.compute_closed_baseline(board_info)
board_info['closed_baseline'] = baseline
print(f"\nBaseline range: {baseline.min():.2f} ~ {baseline.max():.2f}")

matrix, scores = b.analyze_board(board_info)

for r in [0, 1, 2, 7, 8]:
    print(f"\n=== Row {r} logical (cols 14-18) ===")
    for ci in range(14, min(19, cols)):
        print(f"  col {ci}: v={matrix[r,ci]} s={scores[r,ci]:.4f}")

# Save cell crops + template scores
os.makedirs(r'D:\workspace\minesweeper-bot\debug_cells', exist_ok=True)
closed_tmpl = m.templates.get('closed_tile.png')
crop_w = closed_tmpl.shape[1]
crop_h = closed_tmpl.shape[0]
screen = c.get_screenshot()

cells_to_check = [
    (0,15),(0,16),(0,17),(0,18),
    (1,16),(1,17),(1,18),
    (2,15),(2,16),(2,17),(2,18),
    (7,14),(7,15),(7,16),(7,17),(7,18),
    (0,0),(0,1)
]

print("\n=== Per-cell template matching ===")
for r, ci in cells_to_check:
    x = int(win_ox - crop_w/2 + ci * step + 0.5)
    y = int(win_oy - crop_h/2 + r * step + 0.5)
    cell_img = screen[y:y+crop_h, x:x+crop_w]
    fname = f"cell_{r}_{ci}.png"
    cv2.imwrite(f"D:\\workspace\\minesweeper-bot\\debug_cells\\{fname}", cell_img)
    print(f"\nCell ({r},{ci}) at crop ({x},{y}):")
    for name in ['1.png','2.png','3.png','4.png','5.png','flag.png','mine.png','open_blank.png','closed_tile.png']:
        template = m.templates.get(name)
        if template is None: continue
        if cell_img.shape[0] < template.shape[0] or cell_img.shape[1] < template.shape[1]: continue
        res = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        print(f"  {name}: {score:.4f}")

print("\nDone. Images saved to debug_cells/")
