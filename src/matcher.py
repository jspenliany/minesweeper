import cv2
import numpy as np
import logging
import os

class Matcher:
    def __init__(self, assets_path="D:/workspace/minesweeper-bot/assets"):
        self.templates = {}
        self.assets_path = assets_path
        self.load_templates()

    def load_templates(self):
        anchor_dir = os.path.join(self.assets_path, "anchors")
        if os.path.exists(anchor_dir):
            for file in os.listdir(anchor_dir):
                path = os.path.join(anchor_dir, file)
                self.templates[file] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        tile_dir = os.path.join(self.assets_path, "tiles")
        if os.path.exists(tile_dir):
            for file in os.listdir(tile_dir):
                path = os.path.join(tile_dir, file)
                self.templates[file] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        logging.info(f"Loaded {len(self.templates)} templates.")

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

    def match_all(self, target_img, template_name, threshold=0.8):
        template = self.templates.get(template_name)
        if template is None:
            return []
        res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(res >= threshold)
        return [(int(x), int(y)) for x, y in zip(locations[1], locations[0])]

    def match_cell(self, cell_img, template_name):
        template = self.templates.get(template_name)
        if template is None or cell_img.shape[0] < template.shape[0] or cell_img.shape[1] < template.shape[1]:
            return 0.0
        res = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val

    def map_value(self, name):
        name = name.lower()
        if name == 'closed_tile.png': return -1
        if name in ('open_blank.png', 'blank.png'): return -2
        if name == 'mine.png': return 9
        if name == 'flag.png': return 10
        try:
            return int(''.join(filter(str.isdigit, name)))
        except ValueError:
            return -2
