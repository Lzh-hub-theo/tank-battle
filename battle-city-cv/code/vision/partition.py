import numpy as np

class GridPartition:
    
    @staticmethod
    def extract_wall_cells(mask, grid_size, threshold=0.3):
        """
        将mask拆成grid级别的小块（解决连通区域问题）
        """
        h, w = mask.shape
        cells = []

        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                cell = mask[y:y+grid_size, x:x+grid_size]

                if cell.shape[0] != grid_size or cell.shape[1] != grid_size:
                    continue

                ratio = np.sum(cell > 0) / (grid_size * grid_size)

                if ratio > threshold:
                    cells.append((x, y, grid_size, grid_size))

        return cells