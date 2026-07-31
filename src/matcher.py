import cv2
import numpy as np
import logging
import os
from src.timer import timer

class Matcher:
    def __init__(self, assets_path=None):
        if assets_path is None:
            assets_path = os.path.join(os.path.dirname(__file__), "..", "assets")
        assets_path = os.path.abspath(assets_path)
        self.templates = {}
        self.assets_path = assets_path
        self.load_templates()

    @staticmethod
    def _crop_black_border(img, threshold=40):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        h, w = gray.shape
        top = 0
        while top < h and gray[top, :].max() <= threshold:
            top += 1
        bottom = h - 1
        while bottom > top and gray[bottom, :].max() <= threshold:
            bottom -= 1
        left = 0
        while left < w and gray[:, left].max() <= threshold:
            left += 1
        right = w - 1
        while right > left and gray[:, right].max() <= threshold:
            right -= 1
        return img[top:bottom+1, left:right+1]

    def load_templates(self):
        anchor_dir = os.path.join(self.assets_path, "anchors")
        if os.path.exists(anchor_dir):
            for file in os.listdir(anchor_dir):
                path = os.path.join(anchor_dir, file)
                tmpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if tmpl is not None:
                    tmpl = self._crop_black_border(tmpl)
                self.templates[file] = tmpl
        tile_dir = os.path.join(self.assets_path, "tiles")
        if os.path.exists(tile_dir):
            for file in os.listdir(tile_dir):
                path = os.path.join(tile_dir, file)
                self.templates[file] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # Auto-crop number templates to digit-only (remove background noise)
        for name in ("1.png", "2.png", "3.png", "4.png", "5.png", "6.png"):
            tmpl = self.templates.get(name)
            if tmpl is not None:
                digit = self._crop_digit(tmpl)
                if digit is not None:
                    self.templates[name.replace(".png", "_digit.png")] = digit
        logging.info(f"Loaded {len(self.templates)} templates.")

    @staticmethod
    def _crop_digit(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if thresh.mean() > 127:
            thresh = 255 - thresh
        fg = np.where(thresh > 0)
        if len(fg[0]) == 0:
            return None
        y1, y2 = int(fg[0].min()), int(fg[0].max()) + 1
        x1, x2 = int(fg[1].min()), int(fg[1].max()) + 1
        y1 = max(0, y1 - 1)
        y2 = min(img.shape[0], y2 + 1)
        x1 = max(0, x1 - 1)
        x2 = min(img.shape[1], x2 + 1)
        return img[y1:y2, x1:x2]

    @timer
    def find_image(self, target_img, template_name, threshold=0.8):
        template = self.templates.get(template_name)
        if template is None:
            return None
        res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            h, w = template.shape[:2]
            return max_loc[0], max_loc[1], w, h
        return None

    @timer
    def match_all(self, target_img, template_name, threshold=0.8):
        template = self.templates.get(template_name)
        if template is None:
            return []
        res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(res >= threshold)
        return [(int(x), int(y)) for x, y in zip(locations[1], locations[0])]

    @timer
    def match_cell(self, cell_img, template_name):
        template = self.templates.get(template_name)
        if template is None or cell_img.shape[0] < template.shape[0] or cell_img.shape[1] < template.shape[1]:
            return 0.0
        res = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val

    def resize_tile_templates(self, cell_w, cell_h):
        tile_dir = os.path.join(self.assets_path, "tiles")
        for file in os.listdir(tile_dir):
            key = file
            if key in self.templates:
                tmpl = self.templates[key]
                if tmpl.shape[1] == cell_w and tmpl.shape[0] == cell_h:
                    continue
                self.templates[key] = cv2.resize(tmpl, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
        for name in ("1.png", "2.png", "3.png", "4.png", "5.png", "6.png"):
            tmpl = self.templates.get(name)
            if tmpl is not None:
                digit = self._crop_digit(tmpl)
                if digit is not None:
                    self.templates[name.replace(".png", "_digit.png")] = digit
        logging.info(f"Resized tiles to {cell_w}x{cell_h}")

    def map_value(self, name):
        name = name.lower().replace("_digit", "")
        if name == 'closed_tile.png': return -1
        if name in ('open_blank.png', 'blank.png'): return -2
        if name == 'mine.png': return 9
        if name == 'flag.png': return 10
        try:
            return int(''.join(filter(str.isdigit, name)))
        except ValueError:
            return -2
