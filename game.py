import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F



device = torch.device("cuda")



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



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



class Agent:
    def __init__(self, input_dim, n_actions, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, lr=0.001):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(input_dim, n_actions).to(device)
        self.target_model = DQN(input_dim, n_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(10000)

    def predict(self, state):
        if random.random() < self.epsilon:
            # Random action, exploration
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = self.model(state_tensor)
                # Exploitation
                return torch.argmax(q_values).item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Current Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Next Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]

        # Target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss
        loss = self.criterion(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



class Player:
    def __init__(self, x, y, fov_radius, grid_size):
        self.position = [x, y]
        self.fov_radius = fov_radius
        self.grid_size = grid_size
        self.vision = []

    def move(self, direction, walls):
        x, y = self.position
        # UP
        if direction == 0 and x > 0 and walls[x - 1][y] != "w":
            x -= 1
        # DOWN
        elif direction == 1 and x < self.grid_size - 1 and walls[x + 1][y] != "w":
            x += 1
        # LEFT
        elif direction == 2 and y > 0 and walls[x][y - 1] != "w":
            y -= 1
        # RIGHT
        elif direction == 3 and y < self.grid_size - 1 and walls[x][y + 1] != "w":
            y += 1
        # STAY
        elif direction == 4:
            pass
        self.position = [x, y]
        self.update_vision(walls)

    def update_vision(self, walls):
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

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if np.sqrt((x - hx) ** 2 + (y - hy) ** 2) <= self.fov_radius:
                    line = bresenham(hx, hy, x, y)
                    visible = True
                    for lx, ly in line:
                        if walls[lx][ly] == "w":
                            visible = False
                            break
                    if visible:
                        self.vision.append((lx, ly))

    def can_see(self, other_position):
        return tuple(other_position) in self.vision



class Environment:
    def __init__(self, grid_size, turns):
        self.grid_size = grid_size
        self.turns = turns
        self.walls = self.generate_field(grid_size)

        hunter_pos, prey_pos = random.sample(self.accessible_tiles, 2)

        self.hunter = Player(hunter_pos[0], hunter_pos[1], fov_radius=5, grid_size=grid_size)
        self.prey = Player(prey_pos[0], prey_pos[1], fov_radius=5, grid_size=grid_size)

        self.hunter.update_vision(self.walls)
        self.prey.update_vision(self.walls)

    def generate_field(self, size):
        # Modify p_set to set up what percentage [!walls, walls]
        p_set = 1.0
        field = np.random.choice([0, 1], size=(size, size), p=[p_set, 1.0-p_set])
        field[0, :] = 1
        field[-1, :] = 1
        field[:, 0] = 1
        field[:, -1] = 1

        wall_map = np.full((size, size), ".", dtype=str)
        wall_map[field == 1] = "w"

        self.accessible_tiles = [(x, y) for x in range(size) for y in range(size) if wall_map[x][y] == "."]

        while True:
            hunter_pos, prey_pos = random.sample(self.accessible_tiles, 2)
            if self.check_accessibility(wall_map, hunter_pos, prey_pos):
                break

        self.hunter = Player(hunter_pos[0], hunter_pos[1], fov_radius=5, grid_size=size)
        self.prey = Player(prey_pos[0], prey_pos[1], fov_radius=5, grid_size=size)
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
                if 0 <= nx < len(field) and 0 <= ny < len(field[0]) and (nx, ny) not in visited and field[nx][ny] == ".":
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        return False

    def get_state(self):
        hunter_x, hunter_y = self.hunter.position
        prey_x, prey_y = self.prey.position
        hunter_sees_prey = 1 if self.hunter.can_see(self.prey.position) else 0
        prey_sees_hunter = 1 if self.prey.can_see(self.hunter.position) else 0
        dx = abs(hunter_x - prey_x)
        dy = abs(hunter_y - prey_y)
        return np.array([hunter_x, hunter_y, prey_x, prey_y, hunter_sees_prey, prey_sees_hunter, dx, dy], dtype=np.float32)

    def step(self, hunter_action, prey_action):
        self.hunter.move(hunter_action, self.walls)
        self.prey.move(prey_action, self.walls)

        if self.hunter.position == self.prey.position:
            reward_hunter = +10.0
            reward_prey   = -10.0
            done = True
        else:
            reward_hunter = -0.1
            reward_prey   = +0.1
            done = False


        return self.get_state(), reward_hunter, reward_prey, done
    
    
    def render(self):
        grid = [row[:] for row in self.walls]
        hx, hy = self.hunter.position
        px, py = self.prey.position
        grid[hx][hy] = "H"
        grid[px][py] = "P"
        print("\n".join(" ".join(row) for row in grid))
        print("-" * 40)



def train_Environment(episodes, grid_size, turns, batch_size, target_update_interval=10):
    hunter_agent = Agent(input_dim=8, n_actions=5)
    prey_agent = Agent(input_dim=8, n_actions=5)

    rewards_hunter = []
    rewards_prey = []

    for episode in range(episodes):
        env = Environment(grid_size, turns)

        done = False
        total_reward_hunter = 0.0
        total_reward_prey = 0.0
        state = env.get_state()

        print(f"Starting Episode {episode + 1}/{episodes}")
        
        for turn in range(turns):
            os.system("cls" if os.name == "nt" else "clear")
            if done:
                break
            
            # Action pick
            hunter_action = hunter_agent.predict(state)
            prey_action = prey_agent.predict(state)
            # Step settings
            next_state, reward_hunter, reward_prey, done = env.step(hunter_action, 4)

            # Map visualisation
            print(f"Turn: {turn + 1}")
            env.render()
            # Total rewards update
            
            total_reward_hunter += reward_hunter
            total_reward_prey += reward_prey

            print(f"Hunter action: {hunter_action}, Prey action: {prey_action}")
            print(f"Hunter step reward: {reward_hunter:.2f}, Prey step reward: {reward_prey:.2f}")
            print(f"Total hunter reward: {total_reward_hunter:.2f}, total prey reward: {total_reward_prey:.2f}")

            print(f"Episode {episode + 1}/{episodes}, "
                f"Epsilon_hunter: {hunter_agent.epsilon:.2f}, "
                f"Epsilon_prey: {prey_agent.epsilon:.2f}")
            
            # Buffer update
            hunter_agent.replay_buffer.push(state, hunter_action, reward_hunter, next_state, done)
            prey_agent.replay_buffer.push(state, prey_action, reward_prey, next_state, done)


            # Agent trains here
            hunter_agent.train(batch_size)
            prey_agent.train(batch_size)

            state = next_state

        rewards_hunter.append(total_reward_hunter)
        rewards_prey.append(total_reward_prey)

        # Target model update
        if episode % target_update_interval == 0:
            hunter_agent.update_target_model()
            prey_agent.update_target_model()

        # Epsilon update
        hunter_agent.epsilon = max(hunter_agent.epsilon_min, hunter_agent.epsilon * hunter_agent.epsilon_decay)
        prey_agent.epsilon = max(prey_agent.epsilon_min, prey_agent.epsilon * prey_agent.epsilon_decay)

    plt.figure(figsize=(10, 6))

    plt.plot(rewards_hunter, label='Hunter total reward', color='blue', alpha=0.6)
    plt.plot(rewards_prey,   label='Prey total reward',   color='red',  alpha=0.6)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()



train_Environment(episodes=3000, grid_size=8, turns=50, batch_size=32)
