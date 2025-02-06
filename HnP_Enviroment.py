import random
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from matplotlib.animation import PillowWriter
from matplotlib.colors import ListedColormap

from Hnp_Player import *
from HnP_Pathfinding import a_star_distance_modified, a_star_distance_for_hunter


# ------------------- ENVIRONMENT ------------------- #

class Environment:
    def __init__(self, grid_size, turns):
        self.grid_size = grid_size
        self.turns = turns
        
        # Accumulate rewards per episode
        self.cumulative_reward_hunter = 0.0
        self.cumulative_reward_prey = 0.0
        
        # Generating a randomised field
        while True:
            wall_map = self.generate_field(grid_size)
            hunter_pos, prey_pos = self.pick_positions(wall_map)
            if hunter_pos is not None and prey_pos is not None:
                break
                
        self.walls = wall_map
        fov = int(np.sqrt((grid_size ** 2) * 2))
        self.hunter = Player(hunter_pos[0], hunter_pos[1], fov_radius=fov, grid_size=grid_size)
        self.prey = Player(prey_pos[0], prey_pos[1], fov_radius=fov, grid_size=grid_size)
        
    def generate_field(self, size):
        """Generates the playground field with randomized walls

        Args:
            size (int): The size*size parameter for the playground generation

        Returns:
            list: The map grid
        """
        p_set = 0.8
        field = np.random.choice([0, 1], size=(size, size), p=[p_set, 1 - p_set])
        field[0, :] = 1
        field[-1, :] = 1
        field[:, 0] = 1
        field[:, -1] = 1
        wall_map = np.full((size, size), ".", dtype=str)
        wall_map[field == 1] = "w"
        self.accessible_tiles = [(x, y) for x in range(size) for y in range(size) if wall_map[x][y] == "."]
        return wall_map.tolist()
    
    def pick_positions(self, wall_map):
        """Picks a random position on the map and returns Hunter and Prey position
        None if the position does not pass the check_accessibility test

        Args:
            wall_map (list): The playground with walls 

        Returns:
            list, list: If accessible, Hunter position, Prey position. (None, None) if not accesible
        """
        if len(self.accessible_tiles) < 2:
            return None, None
        for attempt in range(100):
            hunter_pos, prey_pos = random.sample(self.accessible_tiles, 2)
            if self.check_accessibility(wall_map, hunter_pos, prey_pos):
                return hunter_pos, prey_pos
        return None, None
    
    def check_accessibility(self, field, start, end):
        queue = [start]
        visited = set()
        while queue:
            x, y = queue.pop(0)
            if (x, y) == end:
                return True
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(field) and 0 <= ny < len(field[0]) and (nx, ny) not in visited and field[nx][ny] == ".":
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        return False
    
    def get_state(self):
        """The state is grouped of:
            - 8 basic features: [hunter_x, hunter_y, prey_x, prey_y, hunter_sees_prey, prey_sees_hunter, dx, dy]
            - The Hunter's local view (a 5x5 patch, 25 elements)
            - The Prey's local view (a 5x5 patch, 25 elements)
        The resulting vector has a size of 8 + 25 + 25 = 58

        Returns:
            ndarray: Current model state
        """
        self.hunter.update_vision(self.walls)
        self.prey.update_vision(self.walls)

        # Get positions
        hunter_x, hunter_y = self.hunter.position
        prey_x, prey_y = self.prey.position

        # For Hunter's perspective:
        if (prey_x, prey_y) in self.hunter.vision:
            hunter_sees_prey = 1
            visible_prey_x = prey_x
            visible_prey_y = prey_y
        else:
            hunter_sees_prey = 0
            visible_prey_x = -1
            visible_prey_y = -1

        # For Prey's perspective (to decide whether to reveal opponent's position):
        if (hunter_x, hunter_y) in self.prey.vision:
            prey_sees_hunter = 1
            visible_hunter_x = hunter_x
            visible_hunter_y = hunter_y
        else:
            prey_sees_hunter = 0
            visible_hunter_x = -1
            visible_hunter_y = -1

        # Compute differences only if both see each other (otherwise set to 0)
        if hunter_sees_prey and prey_sees_hunter:
            dx = hunter_x - prey_x
            dy = hunter_y - prey_y
        else:
            dx = 0
            dy = 0

        # In the global state, we keep Hunter's coordinates always at positions [0,1]
        # and for the opponent, we use the values from Hunter's perspective.
        base_state = np.array([
            hunter_x, hunter_y,               # Hunter's own position (always known)
            visible_prey_x, visible_prey_y,   # Prey's position as seen by Hunter (or -1 if not visible)
            hunter_sees_prey,                 # Flag: Hunter sees Prey
            prey_sees_hunter,                 # Flag: Prey sees Hunter
            dx, dy
        ], dtype=np.float32)

        hunter_patch = self.hunter.get_local_view(self.walls, patch_radius=2)
        prey_patch = self.prey.get_local_view(self.walls, patch_radius=2)
        full_state = np.concatenate([base_state, hunter_patch, prey_patch])
        return full_state
    
    def step(self, hunter_action, prey_action):
        """
        Reward logic with accumulation of shaping rewards:
         - If the Hunter catches the Prey, the final reward for Hunter is 30 + cumulative_reward_hunter,
           for Prey it is -cumulative_reward_prey, and the episode terminates.
         - If not done, Hunter receives a shaping reward based on a_star_distance_for_hunter,
           and Prey receives a shaping reward (the distance computed using a_star_distance_modified with a possible penalty
           if Hunter is within Prey's field of view).
         Both cumulative rewards are clipped to [0, 30].

        Args:
            hunter_action: Predicted action for Hunter.
            prey_action: Predicted action for Prey.

        Returns:
            (reward_hunter, reward_prey, done)
        """
        done = False

        # Hunter step
        # Move Hunter
        self.hunter.move(hunter_action, self.walls)
        
        # Compute shaping reward for Hunter based on quality of view using A* (for Hunter)
        hunter_cost = a_star_distance_for_hunter(self.walls,
                                                 tuple(self.hunter.position),
                                                 tuple(self.prey.position),
                                                 self.grid_size)
        if hunter_cost is not None:
            shaping_reward_hunter = max(0, 30 - hunter_cost)
        else:
            shaping_reward_hunter = 0.0

        self.cumulative_reward_hunter += shaping_reward_hunter
        self.cumulative_reward_hunter = max(0, min(self.cumulative_reward_hunter, 30))

        # Check if catch occurred after Hunter's move
        if self.hunter.position == self.prey.position:
            reward_hunter = 30.0 + self.cumulative_reward_hunter
            reward_prey = -self.cumulative_reward_prey
            done = True
            self.cumulative_reward_hunter = 0.0
            self.cumulative_reward_prey = 0.0
            return reward_hunter, reward_prey, done

        # Prey step
        self.prey.move(prey_action, self.walls)
        if self.hunter.position == self.prey.position:
            reward_hunter = 30.0 + self.cumulative_reward_hunter
            reward_prey = -self.cumulative_reward_prey
            done = True
            self.cumulative_reward_hunter = 0.0
            self.cumulative_reward_prey = 0.0
            return reward_hunter, reward_prey, done

        reward_hunter = 0.0
        dist = a_star_distance_modified(self.walls,
                                        tuple(self.hunter.position),
                                        tuple(self.prey.position),
                                        self.grid_size)
        if dist is not None:
            step_reward_prey = dist
        else:
            step_reward_prey = 0.0

        # If Prey sees Hunter, apply a penalty to Prey's reward
        prey_patch = self.prey.get_local_view(self.walls, patch_radius=2).reshape(5, 5)
        cx, cy = self.prey.position
        rel_x = self.hunter.position[0] - cx + 2
        rel_y = self.hunter.position[1] - cy + 2
        if 0 <= rel_x < 5 and 0 <= rel_y < 5:
            if prey_patch[int(rel_x), int(rel_y)] == 1.0:
                step_reward_prey -= 5.0

        self.cumulative_reward_prey += step_reward_prey
        self.cumulative_reward_prey = max(0, min(self.cumulative_reward_prey, 30))
        
        reward_prey = step_reward_prey
        
        return reward_hunter, reward_prey, done
    
    
    def render(self, return_frame=False):
        """
        Visualization function to output the process as a comprehensive console view.
        It overlays:
          - Hunter and Prey positions,
          - Hunter FOV marked as "h",
          - Prey FOV marked as "p" (or "x" if overlapping with Hunter FOV).
        """
        # Update visions for both players
        self.hunter.update_vision(self.walls)
        self.prey.update_vision(self.walls)
        
        # Build visualization field (copy of walls)
        grid = [row[:] for row in self.walls]
        hx, hy = self.hunter.position
        px, py = self.prey.position
        grid[hx][hy] = "H"
        grid[px][py] = "P"
        
        # Visualize Hunter's FOV with "h"
        for (x, y) in self.hunter.vision:
            if grid[x][y] == ".":
                grid[x][y] = "h"
        
        # Visualize Prey's FOV with "p"
        for (x, y) in self.prey.vision:
            if grid[x][y] == ".":
                grid[x][y] = "p" if grid[x][y] != "h" else "x"
        
        if return_frame:
            return np.array(grid)
        else:
            os.system("cls" if os.name == "nt" else "clear")
            for row in grid:
                print(" ".join(row))
            print("-" * 40)


def save_animation(frames, filename, fps=12):
    """Processes rendered console output into a matplotlib frame

    Args:
        frames (int): Number of frames in the animation
        filename (str): Name of the animation file
        fps (int, optional): FPS settings. Defaults to 12.
    """
    # Remap:
    # "." (clear field) -> 0
    # "w" (wall) -> 1
    # "H" (Hunter) -> 2
    # "P" (Prey) -> 3
    # "h" (Hunter FOV) -> 4
    # "p" (Prey FOV) -> 5
    # "x" (overlaping FOV) -> 6
    mapping = {
        ".": 0,
        "w": 1,
        "H": 2,
        "P": 3,
        "h": 4,
        "p": 5,
        "x": 6
    }
    
    # Custom colors:
    # 0: white, 1: black, 2: red, 3: blue, 4: orange, 5: cyan, 6: magenta
    cmap = ListedColormap(['white', 'black', 'red', 'blue', 'orange', 'cyan', 'magenta'])
    
    # Symbols to numbers
    def frame_to_numeric(frame):
        return np.vectorize(mapping.get)(frame)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    
    # Add a frame
    def update(frame):
        ax.clear()
        ax.axis("off")
        numeric_frame = frame_to_numeric(frame)
        # Используем нашу палитру, значения от 0 до 6
        ax.imshow(numeric_frame, cmap=cmap, vmin=0, vmax=6)
    
    # Stop frames in the end of animation
    if frames:
        frames.extend([frames[-1]] * int(3 * fps))
    
    ani = anime.FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    writer = PillowWriter(fps=fps)
    ani.save(filename, writer=writer)
    print(f"Animation saved as {filename}")
    
    plt.close(fig)