"""Crop number templates to digit-only bounding box and test match scores."""
import cv2
import numpy as np
import os

TILES_DIR = "D:/workspace/minesweeper-bot/assets/tiles"


def auto_crop_digit(img):
    """Find the tight bounding box of the digit using Otsu threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert if the digit is darker than background
    if thresh.mean() > 127:
        thresh = 255 - thresh
    fg = np.where(thresh > 0)
    if len(fg[0]) == 0:
        return None, (0, 0, img.shape[1], img.shape[0])
    y1, y2 = int(fg[0].min()), int(fg[0].max()) + 1
    x1, x2 = int(fg[1].min()), int(fg[1].max()) + 1
    # Add 1px padding
    y1 = max(0, y1 - 1)
    y2 = min(img.shape[0], y2 + 1)
    x1 = max(0, x1 - 1)
    x2 = min(img.shape[1], x2 + 1)
    return img[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


for name in ["1.png", "2.png", "3.png", "4.png", "5.png"]:
    path = os.path.join(TILES_DIR, name)
    t = cv2.imread(path)
    digit, bbox = auto_crop_digit(t)
    if digit is None:
        print(f"{name}: cannot find digit")
        continue
    x, y, w, h = bbox
    print(f"{name}: digit bbox=({x},{y}) {w}x{h} (was {t.shape[1]}x{t.shape[0]})")

    out_name = name.replace(".png", "_digit.png")
    cv2.imwrite(os.path.join(TILES_DIR, out_name), digit)

# Now test: match digit-only vs full-cell templates on debug cells
print("\n--- Match score comparison (full 29x29 vs digit-only) ---")
cell_dir = "D:/workspace/minesweeper-bot/debug_cells"
for cell_file in sorted(os.listdir(cell_dir)):
    if not cell_file.endswith(".png"):
        continue
    cell_path = os.path.join(cell_dir, cell_file)
    cell = cv2.imread(cell_path)

    best_full, best_digit = ("", -1), ("", -1)
    for name in ["1.png", "2.png", "3.png", "4.png", "5.png"]:
        tmpl_full = cv2.imread(os.path.join(TILES_DIR, name))
        tmpl_digit = cv2.imread(os.path.join(TILES_DIR, name.replace(".png", "_digit.png")))

        # Full template with 1px margin (27x27)
        inner_c = cell[1:-1, 1:-1]
        inner_t = tmpl_full[1:-1, 1:-1]
        res = cv2.matchTemplate(inner_c, inner_t, cv2.TM_CCOEFF_NORMED)
        score_full = cv2.minMaxLoc(res)[1]

        # Digit-only: match the smaller digit template inside the cell (no margin)
        inner_c2 = cell[1:-1, 1:-1]  # still remove 1px border
        res = cv2.matchTemplate(inner_c2, tmpl_digit, cv2.TM_CCOEFF_NORMED)
        score_digit = cv2.minMaxLoc(res)[1]

        if score_full > best_full[1]:
            best_full = (name, score_full)
        if score_digit > best_digit[1]:
            best_digit = (name, score_digit)

    print(f"{cell_file}:  full={best_full[0]}({best_full[1]:.3f})  digit={best_digit[0]}({best_digit[1]:.3f})")
