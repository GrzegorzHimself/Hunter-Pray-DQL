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
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------- REPLAY BUFFER ------------------- #
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, map_tensor, scalar_tensor, action, reward, next_map_tensor, next_scalar_tensor, done):
        # Replay buffer setup for shuffle and mini-batch creation
        self.buffer.append((map_tensor, scalar_tensor, action, reward, next_map_tensor, next_scalar_tensor, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        (map_tensors, scalar_tensors, 
         actions, rewards, 
         next_map_tensors, next_scalar_tensors, 
         dones) = zip(*batch)
        
        # Convert them into proper torch tensors
        map_tensors         = torch.stack(map_tensors).to(device)              # shape: [B, 1, grid, grid]
        scalar_tensors      = torch.stack(scalar_tensors).to(device)           # shape: [B, 8]
        next_map_tensors    = torch.stack(next_map_tensors).to(device)         # shape: [B, 1, grid, grid]
        next_scalar_tensors = torch.stack(next_scalar_tensors).to(device)      # shape: [B, 8]

        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones   = torch.tensor(dones,   dtype=torch.float32).to(device)

        return map_tensors, scalar_tensors, actions, rewards, next_map_tensors, next_scalar_tensors, dones

    def __len__(self):
        return len(self.buffer)


# ------------------- DQN NETWORK (Conv2d + MLP) ------------------- #
class ConvDQN(nn.Module):
    def __init__(self, grid_size, scalar_size, n_actions):
        super(ConvDQN, self).__init__()

        # We'll parse a single-channel local_map: shape [B, 1, grid_size, grid_size]
        # + we have "scalar_size" scalar features (like [x_h, y_h, x_p, y_p, sees_h, sees_p, dx, dy])

        # Convolution block (stride=1, padding=1, kernel=3 => shape stays the same)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,  kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1)
        
        # We'll do a "dummy" pass to figure out how many features come out of conv2
        # so that we can safely build our linear layers
        self.grid_size   = grid_size
        self.scalar_size = scalar_size
        self.n_actions   = n_actions

        # We do a dummy forward to compute conv_out_dim:
        with torch.no_grad():
            dummy_map = torch.zeros(1, 1, grid_size, grid_size)  # [batch=1, channel=1, g, g]
            x = self.conv1(dummy_map)
            x = self.conv2(x)
            # x.shape = [1, 16, grid_size, grid_size] if all is correct
            self.conv_out_dim = x.numel()  # Total elements
            # That should be 1 * 16 * grid_size * grid_size

        # We'll combine conv output + scalar_size in a linear layer
        self.fc1 = nn.Linear(self.conv_out_dim + scalar_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, map_tensor, scalar_tensor):
        # map_tensor shape: [B, 1, grid_size, grid_size]
        # scalar_tensor shape: [B, scalar_size]

        x = F.relu(self.conv1(map_tensor))
        x = F.relu(self.conv2(x))

        # Flatten dynamically
        x = x.view(x.size(0), -1)  # shape: [B, conv_out_dim]

        # Concat with scalar input
        x = torch.cat([x, scalar_tensor], dim=1)  # shape: [B, conv_out_dim + scalar_size]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q



# ------------------- AGENT ------------------- #
class Agent:
    def __init__(self, grid_size, scalar_size, n_actions,
                 gamma=0.995, epsilon=1.0,
                 epsilon_decay=0.95, epsilon_min=0.1, lr=0.001):
        self.grid_size    = grid_size
        self.scalar_size  = scalar_size
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_decay= epsilon_decay
        self.epsilon_min  = epsilon_min

        # Build the ConvDQN
        self.model = ConvDQN(grid_size, scalar_size, n_actions).to(device)
        self.target_model = ConvDQN(grid_size, scalar_size, n_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer   = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion   = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(50000)

    def predict(self, map_tensor, scalar_tensor):
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                # Shape for map_tensor => [1, 1, g, g]
                # Shape for scalar_tensor => [1, scalar_size]
                map_tensor    = map_tensor.unsqueeze(0).to(device)
                scalar_tensor = scalar_tensor.unsqueeze(0).to(device)
                q_values      = self.model(map_tensor, scalar_tensor)  # => [1, n_actions]
                action        = torch.argmax(q_values, dim=1).item()
                return action

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        (map_tensors, scalar_tensors,
         actions, rewards,
         next_map_tensors, next_scalar_tensors,
         dones) = self.replay_buffer.sample(batch_size)

        # map_tensors.shape => [B, 1, g, g]
        # scalar_tensors.shape => [B, scalar_size]
        # actions.shape => [B]
        # rewards.shape => [B]
        # next_map_tensors.shape => [B, 1, g, g]
        # next_scalar_tensors.shape => [B, scalar_size]
        # dones.shape => [B]

        # Current Q-values
        q_values = self.model(map_tensors, scalar_tensors)  # shape [B, n_actions]
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Next Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_map_tensors, next_scalar_tensors).max(1)[0]

        # Target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ------------------- PLAYER ------------------- #
class Player:
    def __init__(self, x, y, fov_radius, grid_size):
        self.position   = [x, y]
        self.fov_radius = fov_radius
        self.grid_size  = grid_size
        self.vision     = []
        # local_map: 2D array (grid_size x grid_size),
        #   0=unknown, 1=free cell, 2=wall
        self.local_map  = np.zeros((grid_size, grid_size), dtype=np.int8)

    def move(self, direction, walls):
        # Both Hunter and Prey can move only UP, DOWN, RIGHT, LEFT, and STAY
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
        self.update_vision(walls)

    def update_vision(self, walls):
        # Vision tool using Bresenham's line algorithm
        hx, hy = self.position
        self.vision = []

        def bresenham(x1, y1, x2, y2):
            points = []
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            while True:
                points.append((x1, y1))
                if x1 == x2 and y1 == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
            return points

        for xx in range(self.grid_size):
            for yy in range(self.grid_size):
                dist = np.sqrt((xx - hx)**2 + (yy - hy)**2)
                if dist <= self.fov_radius:
                    line = bresenham(hx, hy, xx, yy)
                    visible = True
                    for lx, ly in line:
                        if walls[lx][ly] == "w":
                            visible = False
                            # Local_map: 2=wall
                            self.local_map[lx, ly] = 2
                            break
                        else:
                            self.local_map[lx, ly] = 1
                    if visible:
                        self.vision.append((xx, yy))

    def can_see(self, other_position):
        return tuple(other_position) in self.vision


# ------------------- ENVIRONMENT ------------------- #
class Environment:
    def __init__(self, grid_size, turns):
        self.grid_size = grid_size
        self.turns     = turns
        self.walls     = self.generate_field(grid_size)

        hunter_pos, prey_pos = random.sample(self.accessible_tiles, 2)

        wall_map = self.generate_field(grid_size)

        while True:
            hunter_pos, prey_pos = random.sample(self.accessible_tiles, 2)
            if self.check_accessibility(wall_map, hunter_pos, prey_pos):
                break

        self.hunter = Player(hunter_pos[0], hunter_pos[1], fov_radius=5, grid_size=grid_size)
        self.prey   = Player(prey_pos[0],   prey_pos[1],   fov_radius=5, grid_size=grid_size)

        self.hunter.update_vision(self.walls)
        self.prey.update_vision(self.walls)

    def generate_field(self, size):
        p_set = 0.8
        field = np.random.choice([0, 1], size=(size, size), p=[p_set, 1 - p_set])
        field[0, :] = 1
        field[-1, :] = 1
        field[:, 0]  = 1
        field[:, -1] = 1

        wall_map = np.full((size, size), ".", dtype=str)
        wall_map[field == 1] = "w"

        self.accessible_tiles = [(x, y) for x in range(size) for y in range(size) if wall_map[x][y] == "."]
        
        return wall_map.tolist()

    def check_accessibility(self, field, start, end):
        queue = [start]
        visited = set()
        while queue:
            x, y = queue.pop(0)
            if (x, y) == end:
                return True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(field) and 0 <= ny < len(field[0]):
                    if (nx, ny) not in visited and field[nx][ny] == ".":
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def get_hunter_state(self):
        hx, hy = self.hunter.position
        px, py = self.prey.position

        # local_map: shape [1, g, g], plus scalar
        map_2d = torch.tensor(self.hunter.local_map, dtype=torch.float32).unsqueeze(0)
        sees_prey   = 1.0 if self.hunter.can_see(self.prey.position) else 0.0
        sees_hunter = 1.0 if self.prey.can_see(self.hunter.position) else 0.0
        dx = hx - px
        dy = hy - py
        scalar = torch.tensor([hx, hy, px, py, sees_prey, sees_hunter, dx, dy], dtype=torch.float32)
        return map_2d, scalar

    def get_prey_state(self):
        hx, hy = self.hunter.position
        px, py = self.prey.position
        map_2d = torch.tensor(self.prey.local_map, dtype=torch.float32).unsqueeze(0)
        sees_hunter = 1.0 if self.prey.can_see(self.hunter.position) else 0.0
        sees_prey   = 1.0 if self.hunter.can_see(self.prey.position) else 0.0
        dx = px - hx
        dy = py - hy
        scalar = torch.tensor([px, py, hx, hy, sees_hunter, sees_prey, dx, dy], dtype=torch.float32)
        return map_2d, scalar

    def step(self, hunter_action, prey_action):
        self.hunter.move(hunter_action, self.walls)
        self.prey.move(prey_action, self.walls)

        if self.hunter.position == self.prey.position:
            reward_hunter = +float(int(self.turns/1.5))
            reward_prey   = -10.0
            done = True
        else:
            reward_hunter = -0.1
            reward_prey   = +0.1
            done = False

        return reward_hunter, reward_prey, done

    def render(self, return_frame=False):
        grid = [row[:] for row in self.walls]
        hx, hy = self.hunter.position
        px, py = self.prey.position
        grid[hx][hy] = "H"
        grid[px][py] = "P"
        if return_frame:
            return np.array(grid)
        else:
            os.system("cls" if os.name == "nt" else "clear")
            print("\n".join(" ".join(row) for row in grid))
            print("-" * 40)



def save_animation(frames, filename, fps=12):
    def frame_to_numeric(frame):
        mapping = {
            ".": 8,   # Empty space
            "w": 0,   # Wall
            "H": 1,   # Hunter
            "P": 2    # Prey
        }
        return np.vectorize(mapping.get)(frame)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")

    # Convert each frame into a grid for animation
    def update(frame):
        ax.clear()
        ax.axis("off")
        numeric_frame = frame_to_numeric(frame)
        ax.imshow(numeric_frame, cmap="gray", vmin=0, vmax=3)

    if frames:
        frames.extend([frames[-1]] * int(3 * fps))

    ani = anime.FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    writer = PillowWriter(fps=fps)
    ani.save(filename, writer = writer)
    print(f"Animation saved as {filename}")



def train_hunter(hunter_agent, prey_agent, episodes, grid_size, turns, batch_size, render_on, n_try):
    rewards_hunter = []
    for episode in range(episodes):
        env = Environment(grid_size, turns)
        h_map, h_scalar = env.get_hunter_state()
        done = False
        total_reward_hunter = 0.0
        frames = []

        for turn in range(turns):
            if done:
                break

            hunter_action = hunter_agent.predict(h_map, h_scalar)

            p_map, p_scalar = env.get_prey_state()
            if len(prey_agent.replay_buffer) < 500:
                prey_action = random.randint(0, 4)
            else:
                prey_action = prey_agent.predict(p_map, p_scalar)

            reward_hunter, _, done = env.step(hunter_action, prey_action)

            if render_on and episode >= episodes - 5:
                    frames.append(env.render(return_frame=True))

            h_map_next, h_scalar_next = env.get_hunter_state()
            hunter_agent.replay_buffer.push(
                h_map, h_scalar, 
                hunter_action, reward_hunter,
                h_map_next, h_scalar_next,
                float(done)
            )
            hunter_agent.train(batch_size)

            h_map, h_scalar = h_map_next, h_scalar_next
            total_reward_hunter += reward_hunter

        hunter_agent.epsilon = max(hunter_agent.epsilon_min, hunter_agent.epsilon * hunter_agent.epsilon_decay)
        if (episode + 1) % 10 == 0:
            hunter_agent.update_target_model()

        rewards_hunter.append(total_reward_hunter)
        print(f"Episode {episode+1} out of {episodes} (Hunter)")

        if render_on and episode >= episodes - 5:
            save_animation(frames, f"hunter_episode_{n_try+1}_{episode+1}.gif")

    return rewards_hunter



def train_prey(prey_agent, hunter_agent, episodes, grid_size, turns, batch_size, render_on, n_try):
    rewards_prey = []
    for episode in range(episodes):
        env = Environment(grid_size, turns)
        p_map, p_scalar = env.get_prey_state()
        done = False
        total_reward_prey = 0.0
        frames = []

        for turn in range(turns):
            if done:
                break

            h_map, h_scalar = env.get_hunter_state()
            hunter_action   = hunter_agent.predict(h_map, h_scalar)

            prey_action = prey_agent.predict(p_map, p_scalar)
            _, reward_prey, done = env.step(hunter_action, prey_action)

            if render_on and episode >= episodes - 5:
                    frames.append(env.render(return_frame=True))


            p_map_next, p_scalar_next = env.get_prey_state()
            prey_agent.replay_buffer.push(
                p_map, p_scalar,
                prey_action, reward_prey,
                p_map_next, p_scalar_next,
                float(done)
            )
            prey_agent.train(batch_size)

            p_map, p_scalar = p_map_next, p_scalar_next
            total_reward_prey += reward_prey

        prey_agent.epsilon = max(prey_agent.epsilon_min, prey_agent.epsilon * prey_agent.epsilon_decay)
        if (episode + 1) % 10 == 0:
            prey_agent.update_target_model()

        rewards_prey.append(total_reward_prey)
        print(f"Episode {episode+1} out of {episodes} (Prey)")

        if render_on and episode >= episodes - 5:
            save_animation(frames, f"prey_episode_{n_try+1}_{episode+1}.gif")

    return rewards_prey


def train_IQL(hunter_agent, prey_agent, episodes_hunter, episodes_prey, grid_size, turns, batch_size, tries, render_on):
    total_reward_hunter = []
    total_reward_prey   = []
    for n_try in range(tries):
        print(f"=== Switching sides! Hunter's turn {n_try+1} ===")
        rh = train_hunter(hunter_agent, prey_agent, episodes_hunter,
                          grid_size, turns, batch_size, render_on, n_try)
        total_reward_hunter.extend(rh)

        print(f"=== Switching sides! Prey's turn {n_try+1} ===")
        rp = train_prey(prey_agent, hunter_agent, episodes_prey,
                        grid_size, turns, batch_size, render_on, n_try)
        total_reward_prey.extend(rp)
    
    plt.close('all')
    # MatPlotLib graphic output of the training cycle conducted
    plt.figure(figsize=(10, 5))
    # Plot Hunter rewards
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(total_reward_hunter)), total_reward_hunter, label="Hunter", color='#0E0598', s=10)
    avg_hunter = [np.mean(total_reward_hunter[max(0, i-50):i+1]) for i in range(len(total_reward_hunter))]
    plt.plot(range(len(total_reward_hunter)), avg_hunter, color='orange', label="Hunter Avg (50)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"IQL Hunter:\nGrid {grid_size}x{grid_size}; Turns {turns}; Episodes {episodes_hunter*tries*2}; FOV 5")
    plt.legend()
    plt.grid(True)
    # Plot Prey rewards
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(total_reward_prey)), total_reward_prey, label="Prey", color='xkcd:baby poop green', s=10)
    rolling_avg_prey = [np.mean(total_reward_prey[max(0, i-50):i+1]) for i in range(len(total_reward_prey))]
    plt.plot(range(len(total_reward_prey)), rolling_avg_prey, color='green', label="Prey Avg (50)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"IQL Prey:\nGrid {grid_size}x{grid_size}; Turns {turns}; Episodes {episodes_hunter*tries*2}; FOV 5")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # === SETTINGS === #
    #
    # grid_size                      -- set up the grid size of the map.
    #                                   Do not compensate for walls (+2), they are not accounted for in the logic
    # episodes_hunter, episodes_prey -- Setup Hunter and Prey episodes for their unique 
    #                                   trainings (SHOULD BE EQUAL)
    # turns                          -- accounted for automatically
    # batch size                     -- setup the batch size
    # tries                          -- the number of cycles both of the models train
    # render_on                      -- if you want to turn the render of the field on to see the process of training
    #                                   May significantly impact the speed of the training
    grid_size = 12
    hunter_agent = Agent(grid_size=grid_size, scalar_size=8, n_actions=5)
    prey_agent   = Agent(grid_size=grid_size, scalar_size=8, n_actions=5)

    train_IQL(hunter_agent, prey_agent,
              episodes_hunter=1000,
              episodes_prey=1000,
              grid_size=grid_size,
              turns=int(grid_size*grid_size*2),
              batch_size=32,
              tries=3,
              render_on=True)
