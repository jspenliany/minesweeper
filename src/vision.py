import cv2
import numpy as np
import pyautogui
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Vision:
    def __init__(self, assets_path="D:/workspace/minesweeper-bot/assets"):
        self.assets_path = assets_path
        self.templates = {}  # Store templates: {name: image_array}
        self.load_templates()

    def load_templates(self):
        """Load all images from assets/anchors and assets/tiles"""
        # Load anchors
        anchor_dir = os.path.join(self.assets_path, "anchors")
        if os.path.exists(anchor_dir):
            for file in os.listdir(anchor_dir):
                path = os.path.join(anchor_dir, file)
                self.templates[file] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        # Load tiles
        tile_dir = os.path.join(self.assets_path, "tiles")
        if os.path.exists(tile_dir):
            for file in os.listdir(tile_dir):
                path = os.path.join(tile_dir, file)
                self.templates[file] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        logging.info(f"Loaded {len(self.templates)} templates.")

    def get_screenshot(self):
        """Capture screen and convert to OpenCV format (BGR)"""
        screenshot = pyautogui.screenshot()
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def find_image(self, target_img, template_name, threshold=0.8):
        """
        Find the first occurrence of a template in the target image
        :return: (x, y, w, h) of the match, or None
        """
        template = self.templates.get(template_name)
        if template is None:
            # Silently return None if template is missing to avoid log flooding
            return None

        res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val >= threshold:
            h, w = template.shape[:2]
            return max_loc[0], max_loc[1], w, h
        return None

    def calibrate_grid(self):
        """
        Automatically find game board boundary and cell size using anchors.
        Finds the anchor, then searches for the first actual closed tile to set as origin.
        """
        screen = self.get_screenshot()
        
        # 1. Find board top-left anchor to get a search region
        anchor = self.find_image(screen, "board_tl.png")
        if not anchor:
            logging.error("Could not find board top-left anchor. Please check assets/anchors/board_tl.png")
            return None
        
        ax, ay, aw, ah = anchor
        
        # 2. Determine cell size using the template
        tile_template = self.templates.get("closed_tile.png")
        if tile_template is None:
            logging.error("closed_tile.png missing in assets.")
            return None
        tw, th = tile_template.shape[1], tile_template.shape[0]
        
        # 3. Search for the FIRST actual closed tile in the vicinity of the anchor
        # We look in a small region to the right and below the anchor
        search_region = screen[ay : ay + 200, ax : ax + 200]
        res = cv2.matchTemplate(search_region, tile_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val < 0.8:
            logging.error("Found anchor but couldn't find any closed tiles nearby. Check your tiles/closed_tile.png")
            return None
            
        # The actual origin is the center of this first found tile
        origin_x = ax + max_loc[0] + tw // 2
        origin_y = ay + max_loc[1] + th // 2
        
        logging.info(f"Grid calibrated: Origin({origin_x}, {origin_y}), CellSize({tw}x{th})")
        return {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "cell_w": tw,
            "cell_h": th
        }

    def analyze_board(self, board_info):
        """
        Analyze the grid and return a logical matrix.
        board_info: {origin_x, origin_y, cell_w, cell_h, rows, cols}
        """
        screen = self.get_screenshot()
        rows = board_info.get('rows', 9) # Default to 9
        cols = board_info.get('cols', 9) # Default to 9
        
        matrix = np.full((rows, cols), -1, dtype=int) # -1: Unknown, 0-8: Numbers, 9: Mine, 10: Flag
        
        for r in range(rows):
            for c in range(cols):
                x = board_info['origin_x'] + (c * board_info['cell_w']) - board_info['cell_w'] // 2
                y = board_info['origin_y'] + (r * board_info['cell_h']) - board_info['cell_h'] // 2
                
                # Crop the cell
                cell_img = screen[y : y + board_info['cell_h'], x : x + board_info['cell_w']]
                
                # Match against tile templates
                for name, template in self.templates.items():
                    # Only match files in assets/tiles
                    if not name.endswith('.png'): continue
                    
                    # Simple template match for the small cell image
                    res = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    
                    if max_val > 0.9: # High threshold for cell internal match
                        # Map filename (e.g., '1.png', 'mine.png') to value
                        val = self._map_template_to_value(name)
                        matrix[r, c] = val
                        break
        
        return matrix

    def _map_template_to_value(self, name):
        """Helper to convert filename to matrix value"""
        name = name.lower()
        if name == 'closed_tile.png': return -1
        if name == 'mine.png': return 9
        if name == 'flag.png': return 10
        try:
            return int(''.join(filter(str.isdigit, name)))
        except ValueError:
            return -1
