import numpy as np
import math

from HnP_Bresenham import *

# ------------------- PLAYER ------------------- #
class Player:
    def __init__(self, x, y, fov_radius, grid_size):
        self.position = [x, y]
        self.fov_radius = fov_radius
        self.grid_size = grid_size
        self.vision = []  # Visible cells
        
    def move(self, direction, walls):
        """Ruleset of moves

        Args:
            direction (int): Predicted move
            walls (list): A list of walls coordinates
        """
        x, y = self.position
        if direction == 0 and x > 0 and walls[x - 1][y] != "w":
            x -= 1      # UP
        elif direction == 1 and x < self.grid_size - 1 and walls[x + 1][y] != "w":
            x += 1      # DOWN
        elif direction == 2 and y > 0 and walls[x][y - 1] != "w":
            y -= 1      # LEFT
        elif direction == 3 and y < self.grid_size - 1 and walls[x][y + 1] != "w":
            y += 1      # RIGHT
        elif direction == 4:
            pass        # STAY
        self.position = [x, y]
        
    def get_local_view(self, walls, patch_radius=2):
        """Returns local patch size of (2*patch_radius+1)x(2*patch_radius+1)

        Args:
            walls (list): A list of walls coordinates
            patch_radius (int, optional): Radius of cells the Agent knows about. Defaults to 2.

        Returns:
            ndarray: Information about surrounding cells
        """
        patch_size = 2 * patch_radius + 1
        local_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        cx, cy = self.position
        for dx in range(-patch_radius, patch_radius+1):
            for dy in range(-patch_radius, patch_radius+1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    local_patch[dx + patch_radius, dy + patch_radius] = 1.0 if walls[x][y] != "w" else 0.0
                else:
                    local_patch[dx + patch_radius, dy + patch_radius] = 0.0
        return local_patch.flatten()
    
    def update_vision(self, walls):
        """Updates the vision of an object
        Uses Bresenham Supercover algorithm

        Args:
            walls (list): A list of walls coordinates
        """
        self.vision = []
        hx, hy = self.position
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Check if the  cell is within fov_radius
                if math.sqrt((x - hx)**2 + (y - hy)**2) <= self.fov_radius:
                    line = bresenham(hx, hy, x, y)
                    visible = True
                    for lx, ly in line:
                        if walls[lx][ly] == "w":
                            visible = False
                            break
                    if visible:
                        self.vision.append((x, y))
