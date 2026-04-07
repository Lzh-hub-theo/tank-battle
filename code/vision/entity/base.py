import numpy as np

class BaseDetector:
    def __init__(self):
        pass

    # 指定位置生成掩膜
    def get_mask(self, hsv):
        height, width = hsv.shape[:2]
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        y1, x1 = 387, 198
        y2, x2 = 415, 228
        
        mask[y1:y2, x1:x2] = 255
        
        return mask