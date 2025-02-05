import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from matplotlib.animation import PillowWriter
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
import heapq
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------- Bresenham Supercover ------------------- #
def bresenham(x1, y1, x2, y2):
    """
    A 'supercover' version of Bresenham's line algorithm that ensures all
    cells touched by the line are included.
    """
    points = []
    # Record the first point
    points.append((x1, y1))
    
    dx = x2 - x1
    dy = y2 - y1
    
    # Determine directions of steps
    xstep = 1 if dx >= 0 else -1
    ystep = 1 if dy >= 0 else -1
    
    dx = abs(dx)
    dy = abs(dy)
    
    # Double increments for dx, dy
    ddx = 2 * dx
    ddy = 2 * dy
    
    x, y = x1, y1  # start at first cell
    
    if dx >= dy:
        errorprev = error = dx  # start error in the middle
        for i in range(dx):
            x += xstep
            error += ddy
            if error > ddx:
                y += ystep
                error -= ddx
                if error + errorprev < ddx:
                    points.append((x, y - ystep))
                elif error + errorprev > ddx:
                    points.append((x - xstep, y))
                else:
                    points.append((x, y - ystep))
                    points.append((x - xstep, y))
            points.append((x, y))
            errorprev = error
    else:
        errorprev = error = dy
        for i in range(dy):
            y += ystep
            error += ddx
            if error > ddy:
                x += xstep
                error -= ddy
                if error + errorprev < ddy:
                    points.append((x - xstep, y))
                elif error + errorprev > ddy:
                    points.append((x, y - ystep))
                else:
                    points.append((x - xstep, y))
                    points.append((x, y - ystep))
            points.append((x, y))
            errorprev = error
            
    return points


# ------------------- REPLAY BUFFER ------------------- #
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device),
        )
        
    def __len__(self):
        return len(self.buffer)


# ------------------- RNN-DQN NETWORK ------------------- #
class RNN_DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, num_layers=1):
        """
        input_dim: input vector size (ex, 58)
        hidden_dim: hidden LSTM state
        n_actions: number of actions (wx, 5)
        num_layers: number of LSTM layers
        """
        super(RNN_DQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_actions)
        
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)  # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]                # use the output of the last time step
        q_values = self.fc(out)
        return q_values, hidden


# ------------------- RNN-based AGENT ------------------- #
class RNNAgent:
    def __init__(self, input_dim, n_actions, hidden_dim=128, num_layers=1,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, lr=0.001):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.model = RNN_DQN(input_dim, hidden_dim, n_actions, num_layers).to(device)
        self.target_model = RNN_DQN(input_dim, hidden_dim, n_actions, num_layers).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(10000)
        
    def predict(self, state, hidden=None):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                q_values, _ = self.model(state_tensor, hidden)
                return torch.argmax(q_values, dim=1).item()
            
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.unsqueeze(1)       # [batch, seq_len=1, input_dim]
        next_states = next_states.unsqueeze(1)
        q_values, _ = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q_values, _ = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ------------------- PLAYER ------------------- #
class Player:
    def __init__(self, x, y, fov_radius, grid_size):
        self.position = [x, y]
        self.fov_radius = fov_radius
        self.grid_size = grid_size
        self.vision = []  # Visible tiles
        
    def move(self, direction, walls):
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
        """
        Returns local patch size of (2*patch_radius+1)x(2*patch_radius+1)
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
        """
        Updates the vision of an object
        Uses Bresenham Supercover algorithm
        """
        self.vision = []
        hx, hy = self.position
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Check if the etile is within fov_radius
                if math.sqrt((x - hx)**2 + (y - hy)**2) <= self.fov_radius:
                    line = bresenham(hx, hy, x, y)
                    visible = True
                    for lx, ly in line:
                        if walls[lx][ly] == "w":
                            visible = False
                            break
                    if visible:
                        self.vision.append((x, y))


# ------------------- Pathfinding ------------------- #
def cell_cost_hunter(x, y, walls, grid_size):
    """
    Calculates the cost of moving to (x, y)
    The sampling patch 3x3
    """
    base_cost = 1.0
    patch_radius = 1
    free_count = 0
    count = 0
    for dx in range(-patch_radius, patch_radius + 1):
        for dy in range(-patch_radius, patch_radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                count += 1
                if walls[nx][ny] != "w":
                    free_count += 1
    view_quality = free_count / count  # доля свободных клеток (максимум 1.0)
    beta = 0.5  # коэффициент влияния обзора
    multiplier = 1 - beta * view_quality
    cost = base_cost * multiplier
    return cost

def a_star_distance_for_hunter(walls, start, goal, grid_size):
    """
    Modified A* for Hunter, calculating how open the chosen cell is
    """
    if start == goal:
        return 0
    (sx, sy) = start
    (gx, gy) = goal
    if walls[sx][sy] == "w" or walls[gx][gy] == "w":
        return None
    open_set = []
    heapq.heappush(open_set, (0, sx, sy))
    cost_so_far = {(sx, sy): 0}
    
    def heuristic(ax, ay, bx, by):
        return abs(ax - bx) + abs(ay - by)
    
    while open_set:
        priority, cx, cy = heapq.heappop(open_set)
        if (cx, cy) == (gx, gy):
            return cost_so_far[(cx, cy)]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and walls[nx][ny] == ".":
                new_cost = cost_so_far[(cx, cy)] + cell_cost_hunter(nx, ny, walls, grid_size)
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    prio = new_cost + heuristic(nx, ny, gx, gy)
                    heapq.heappush(open_set, (prio, nx, ny))
    return None

def a_star_distance_modified(walls, start, goal, grid_size):
    """
    Modified A* for Prey
    Cell cost set to 1, or lower if there is a wall nearby
    Prefer lower cost
    """
    if start == goal:
        return 0
    (sx, sy) = start
    (gx, gy) = goal
    if walls[sx][sy] == "w" or walls[gx][gy] == "w":
        return None
    open_set = []
    heapq.heappush(open_set, (0, sx, sy))
    cost_so_far = {(sx, sy): 0}
    
    def heuristic(ax, ay, bx, by):
        return abs(ax - bx) + abs(ay - by)
    
    def cell_cost(x, y):
        cost = 1.0
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if walls[nx][ny] == "w":
                    cost = 0.8
                    break
        return cost
    
    while open_set:
        priority, cx, cy = heapq.heappop(open_set)
        if (cx, cy) == (gx, gy):
            return cost_so_far[(cx, cy)]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and walls[nx][ny] == ".":
                new_cost = cost_so_far[(cx, cy)] + cell_cost(nx, ny)
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    prio = new_cost + heuristic(nx, ny, gx, gy)
                    heapq.heappush(open_set, (prio, nx, ny))
    return None


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
        """
        The state is grouped of:
         - 8 basic features: [hunter_x, hunter_y, prey_x, prey_y, hunter_sees_prey, prey_sees_hunter, dx, dy]
         - The Hunter's local view (a 5x5 patch, 25 elements)
         - The Prey's local view (a 5x5 patch, 25 elements)
        The resulting vector has a size of 8 + 25 + 25 = 58 features
        """
        self.hunter.update_vision(self.walls)
        self.prey.update_vision(self.walls)
        
        hunter_x, hunter_y = self.hunter.position
        prey_x, prey_y = self.prey.position
        
        # Does Hunter sees Prey
        if (prey_x, prey_y) in self.hunter.vision:
            hunter_sees_prey = 1
            visible_prey_x = prey_x
            visible_prey_y = prey_y
        else:
            hunter_sees_prey = 0
            visible_prey_x = -1
            visible_prey_y = -1

        # Does Prey sees Hunter
        if (hunter_x, hunter_y) in self.prey.vision:
            prey_sees_hunter = 1
        else:
            prey_sees_hunter = 0

        # If at least one edoesn't see, the difference will be 0
        if hunter_sees_prey and prey_sees_hunter:
            dx = hunter_x - prey_x
            dy = hunter_y - prey_y
        else:
            dx = 0
            dy = 0

        base_state = np.array([hunter_x, hunter_y, visible_prey_x, visible_prey_y,
                            hunter_sees_prey, prey_sees_hunter, dx, dy], dtype=np.float32)
        hunter_patch = self.hunter.get_local_view(self.walls, patch_radius=2)
        prey_patch = self.prey.get_local_view(self.walls, patch_radius=2)
        full_state = np.concatenate([base_state, hunter_patch, prey_patch])
        return full_state
    
    def step(self, hunter_action, prey_action):
        """
        Reward logic with the accumulation of shaping rewards:
         -  If the hunter catches the prey, the final reward for the hunter is +30 + cumulative_reward,
            for the prey it is -cumulative_reward, and the episode terminates
         -  If not done, Hunter receives 0, and Prey receives a shaping reward
            (the distance computed using a_star_distance_modified with a possible penalty if Hunter is within Prey's field of view)
        """
        done = False

        # Hunter step
        self.hunter.move(hunter_action, self.walls)
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
        dist = a_star_distance_modified(self.walls, tuple(self.hunter.position),
                                        tuple(self.prey.position), self.grid_size)
        if dist is not None:
            step_reward_prey = dist
        else:
            step_reward_prey = 0.0

        # If Prey sees Hunter => punish Prey:
        prey_patch = self.prey.get_local_view(self.walls, patch_radius=2).reshape(5, 5)
        cx, cy = self.prey.position
        rel_x = self.hunter.position[0] - cx + 2
        rel_y = self.hunter.position[1] - cy + 2
        if 0 <= rel_x < 5 and 0 <= rel_y < 5:
            if prey_patch[int(rel_x), int(rel_y)] == 1.0:
                step_reward_prey -= 5.0

        # Normalise Prey reward to [0, 30]:
        step_reward_prey = max(0, min(step_reward_prey, 30))

        self.cumulative_reward_prey += step_reward_prey
        reward_prey = step_reward_prey
        
        return reward_hunter, reward_prey, done
    
    def render(self, return_frame=False):
        # Before renderind, update visualisation for both players
        self.hunter.update_vision(self.walls)
        self.prey.update_vision(self.walls)
        
        # Build visualisation field
        grid = [row[:] for row in self.walls]
        # Map the players
        hx, hy = self.hunter.position
        px, py = self.prey.position
        grid[hx][hy] = "H"
        grid[px][py] = "P"
        
        # Visualise Hunter FOV as "p"
        for (x, y) in self.hunter.vision:
            if grid[x][y] == ".":
                grid[x][y] = "h"
        
        # Visualise Prey FOV as "p"
        for (x, y) in self.prey.vision:
            if grid[x][y] == ".":
                # If FOVs overlap, mark as "x"
                grid[x][y] = "p" if grid[x][y] != "h" else "x"
        
        if return_frame:
            return np.array(grid)
        else:
            os.system("cls" if os.name == "nt" else "clear")
            for row in grid:
                print(" ".join(row))
            print("-" * 40)


def save_animation(frames, filename, fps=12):
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


# ------------------- TRAINING ------------------- #
def train_hunter(hunter_agent, prey_agent, episodes, grid_size, turns, batch_size, render_on, n_try):
    rewards_hunter = []
    for episode in range(episodes):
        env = Environment(grid_size, turns)
        state = env.get_state()
        done = False
        total_reward_hunter = 0.0
        frames = []
        for turn in range(turns):
            if done:
                break
            hunter_action = hunter_agent.predict(state)
            if len(prey_agent.replay_buffer) < 500:
                prey_action = random.randint(0, 4)
            else:
                prey_action = prey_agent.predict(state)
            reward_hunter, _, done = env.step(hunter_action, prey_action)
            next_state = env.get_state()
            if render_on and episode >= episodes - 5:
                frames.append(env.render(return_frame=True))
            hunter_agent.replay_buffer.push(state, hunter_action, reward_hunter, next_state, done)
            hunter_agent.train(batch_size)
            state = next_state
            total_reward_hunter += reward_hunter
        hunter_agent.epsilon = max(hunter_agent.epsilon_min, hunter_agent.epsilon * hunter_agent.epsilon_decay)
        if (episode + 1) % 10 == 0:
            hunter_agent.update_target_model()
        rewards_hunter.append(total_reward_hunter)
        print(f"Episode {episode+1} out of {episodes} (Hunter {n_try+1})")
        if render_on and episode >= episodes - 5:
            save_animation(frames, f"hunter_episode_{n_try+1}_{episode+1}.gif")
    return rewards_hunter

def train_prey(prey_agent, hunter_agent, episodes, grid_size, turns, batch_size, render_on, n_try):
    rewards_prey = []
    for episode in range(episodes):
        env = Environment(grid_size, turns)
        state = env.get_state()
        done = False
        total_reward_prey = 0.0
        frames = []
        for turn in range(turns):
            if done:
                break
            hunter_action = hunter_agent.predict(state)
            prey_action = prey_agent.predict(state)
            _, reward_prey, done = env.step(hunter_action, prey_action)
            next_state = env.get_state()
            if render_on and episode >= episodes - 5:
                frames.append(env.render(return_frame=True))
            prey_agent.replay_buffer.push(state, prey_action, reward_prey, next_state, done)
            prey_agent.train(batch_size)
            state = next_state
            total_reward_prey += reward_prey
        prey_agent.epsilon = max(prey_agent.epsilon_min, prey_agent.epsilon * prey_agent.epsilon_decay)
        if (episode + 1) % 10 == 0:
            prey_agent.update_target_model()
        rewards_prey.append(total_reward_prey)
        print(f"Episode {episode+1} out of {episodes} (Prey {n_try+1})")
        if render_on and episode >= episodes - 5:
            save_animation(frames, f"prey_episode_{n_try+1}_{episode+1}.gif")
    return rewards_prey

def train_IQL(hunter_agent, prey_agent, episodes_hunter, episodes_prey, grid_size, turns, batch_size, tries, render_on):
    total_reward_hunter = []
    total_reward_prey = []
    for n_try in range(tries):
        print(f"=== Switching sides! Hunter's turn {n_try+1} ===")
        rewards_hunter = train_hunter(hunter_agent, prey_agent, episodes_hunter, grid_size, turns, batch_size, render_on, n_try)
        total_reward_hunter.extend(rewards_hunter)
        print(f"=== Switching sides! Prey's turn {n_try+1}! ===")
        rewards_prey = train_prey(prey_agent, hunter_agent, episodes_prey, grid_size, turns, batch_size, render_on, n_try)
        total_reward_prey.extend(rewards_prey)
    
    plt.close('all')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(total_reward_hunter)), total_reward_hunter, label="Hunter", color='#0E0598', s=5)
    avg_hunter = [np.mean(total_reward_hunter[max(0, i-50):i+1]) for i in range(len(total_reward_hunter))]
    plt.plot(range(len(total_reward_hunter)), avg_hunter, color='orange', label="Hunter Avg (50)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"IQL Hunter:\nGrid {grid_size}x{grid_size}; Turns {turns}; Episodes {episodes_hunter*tries*2}")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(total_reward_prey)), total_reward_prey, label="Prey", color='xkcd:baby poop green', s=5)
    rolling_avg_prey = [np.mean(total_reward_prey[max(0, i-50):i+1]) for i in range(len(total_reward_prey))]
    plt.plot(range(len(total_reward_prey)), rolling_avg_prey, color='green', label="Prey Avg (50)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"IQL Prey:\nGrid {grid_size}x{grid_size}; Turns {turns}; Episodes {episodes_hunter*tries*2}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # The input vector size: 8 base states + 25 (Hunter) + 25 (Prey) = 58
    grid_size = 12
    input_dim = 8 + 25*2
    hunter_agent = RNNAgent(input_dim=input_dim, n_actions=5, hidden_dim=128, num_layers=1)
    prey_agent = RNNAgent(input_dim=input_dim, n_actions=5, hidden_dim=128, num_layers=1)
    
    train_IQL(hunter_agent, prey_agent,
              episodes_hunter=500,
              episodes_prey=500,
              grid_size=grid_size,
              turns=int(grid_size * grid_size * 3),
              batch_size=64,
              tries=6,
              render_on=True)
