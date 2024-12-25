import random
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state_vector = self._state_to_vector(state)
        next_state_vector = self._state_to_vector(next_state)
        self.buffer.append((state_vector, action, reward, next_state_vector, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device),
        )

    def __len__(self):
        return len(self.buffer)

    def _state_to_vector(self, state):
        hunter_x, hunter_y = state["hunter_pos"]
        pray_x, pray_y = state["pray_pos"]
        hunter_sees_pray = 1 if tuple(state["pray_pos"]) in state["hunter_vision"] else 0
        pray_sees_hunter = 1 if tuple(state["hunter_pos"]) in state["pray_vision"] else 0
        return [
            hunter_x, hunter_y, pray_x, pray_y,
            abs(hunter_x - pray_x), abs(hunter_y - pray_y),
            hunter_sees_pray, pray_sees_hunter
        ]


class Player:
    def __init__(self, x, y, fov_radius, grid_size):
        self.position = [x, y]
        self.fov_radius = fov_radius
        self.original_fov_radius = fov_radius
        self.grid_size = grid_size
        self.vision = []
        self.turns_stayed = 0

    def is_hunter(self):
        return isinstance(self, Hunter)

    def move(self, direction, walls):
        x, y = self.position
        """if direction == "STAY":
            if self.is_hunter():
                self.turns_stayed += 1
                if self.turns_stayed != 0:
# nice
                    self.fov_radius = self.original_fov_radius + 40
        else:
            self.turns_stayed = 0
            self.fov_radius = self.original_fov_radius"""

        if direction == "UP" and x > 0 and walls[x-1][y] != "w":
            x -= 1
        if direction == "DOWN" and x < self.grid_size - 1 and walls[x+1][y] != "w":
            x += 1
        if direction == "LEFT" and y > 0 and walls[x][y-1] != "w":
            y -= 1
        if direction == "RIGHT" and y < self.grid_size - 1 and walls[x][y+1] != "w":
            y += 1
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
                sqrtr = np.sqrt((x - hx)**2 + (y - hy)**2)
                if sqrtr <= self.fov_radius:
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
    


class Hunter(Player):
    def __init__(self, x, y, fov_radius=None, grid_size=None):
        if grid_size is None:
            raise ValueError("grid_size must be provided")
        super().__init__(x, y, fov_radius, grid_size)


class Pray(Player):
    def __init__(self, x, y, fov_radius=None, grid_size=None):
        if grid_size is None:
            raise ValueError("grid_size must be provided")
        super().__init__(x, y, fov_radius, grid_size)


class Game:
    def __init__(self, grid_size, turns):
        self.grid_size = grid_size
        self.turns = turns
        self.walls = self.generate_field(grid_size)

        hunter_pos, pray_pos = random.sample(self.accessible_tiles, 2)

        self.hunter = Hunter(hunter_pos[0], hunter_pos[1], fov_radius=5, grid_size=grid_size)
        self.pray = Pray(pray_pos[0], pray_pos[1], fov_radius=5, grid_size=grid_size)

        self.hunter.update_vision(self.walls)
        self.pray.update_vision(self.walls)

    def generate_field(self, size):
        field = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])

        field[0, :] = 1
        field[-1, :] = 1
        field[:, 0] = 1
        field[:, -1] = 1

        for _ in range(4):
            new_field = field.copy()
            for x in range(1, size - 1):
                for y in range(1, size - 1):
                    neighbors = np.sum(field[x-1:x+2, y-1:y+2]) - field[x, y]
                    if field[x, y] == 1:
                        if neighbors < 3:
                            new_field[x, y] = 0
                    else:
                        if neighbors > 4:
                            new_field[x, y] = 1
            field = new_field

        wall_map = np.full((size, size), ".", dtype=str)
        wall_map[field == 1] = "w"

        self.accessible_tiles = [(x, y) for x in range(size) for y in range(size) if wall_map[x][y] == "."]
        return wall_map.tolist()

    def get_state(self):
        return {
            "hunter_pos": self.hunter.position,
            "pray_pos": self.pray.position,
            "hunter_vision": self.hunter.vision,
            "pray_vision": self.pray.vision
        }

    def step(self, hunter_action, pray_action, turn):
        self.hunter.move(hunter_action, self.walls)
        self.pray.move(pray_action, self.walls)

        hunter_sees_pray = self.hunter.can_see(self.pray.position)
        pray_sees_hunter = self.pray.can_see(self.hunter.position)

        """if hunter_sees_pray and self.hunter.turns_stayed < 2:
            print(f"Hunter sees the Pray at {self.pray.position}")
            if self.hunter.turns_stayed >= 2:
                print("Enemy UAV inbound!")"""

        """if pray_sees_hunter:
            print(f"Pray sees the Hunter at {self.hunter.position}")"""

        reward_hunter = 0
        reward_pray = 0

        if hunter_sees_pray and self.hunter.turns_stayed == 0:
            reward_hunter += 5
            reward_pray -= 5

        if self.hunter.position == self.pray.position:
            reward_hunter += 50
            reward_pray -= 50
            return self.get_state(), reward_hunter, reward_pray, True

        if not hunter_sees_pray:
            hx, hy = self.hunter.position
            px, py = self.pray.position
            distance = abs(hx - px) + abs(hy - py)
            reward_pray += min(distance * 0.1, 5)
        
        if not hunter_sees_pray and turn % 50 == 0 and self.hunter.position != self.pray.position:
            reward_pray += 10

        reward_hunter -= 0.5
        reward_pray += 0.5

        return self.get_state(), reward_hunter, reward_pray, False

    def render_field(self):
        grid = [row[:] for row in self.walls]

        for x, y in self.hunter.vision:
            if grid[x][y] == ".":
                grid[x][y] = "f"

        for x, y in self.pray.vision:
            if grid[x][y] == "f":
                grid[x][y] = "f"
            elif grid[x][y] == ".":
                grid[x][y] = "f"

        hx, hy = self.hunter.position
        px, py = self.pray.position
        grid[hx][hy] = "h"
        grid[px][py] = "p"

        for row in grid:
            print(" ".join(row))
        print()


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class BaseModel:
    def __init__(self, input_dim, output_dim, actions):
        self.model = DQN(n_observations=input_dim, n_actions=output_dim).to(device)
        
        self.target_model = DQN(n_observations=input_dim, n_actions=output_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.actions = actions
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.epsilon = 0.9
        self.epsilon_decay = 1000
        self.epsilon_min = 0.05

    def get_valid_actions(self, state, walls):
        x, y = state["hunter_pos"] if hasattr(self, "hunter") else state["pray_pos"]
        valid_actions = []
        if x > 0 and walls[x-1][y] != "w":
            valid_actions.append("UP")
        if x < len(walls) - 1 and walls[x+1][y] != "w":
            valid_actions.append("DOWN")
        if y > 0 and walls[x][y-1] != "w":
            valid_actions.append("LEFT")
        if y < len(walls[0]) - 1 and walls[x][y+1] != "w":
            valid_actions.append("RIGHT")
        valid_actions.append("STAY")
        return valid_actions

    def predict(self, state, walls, maximize=True):
        valid_actions = self.get_valid_actions(state, walls)
        if not valid_actions:
            valid_actions.append("STAY")
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state).to(device)
            q_values = self.model(state_tensor)
            action_q_values = {action: q_values[0][self.actions.index(action)] for action in valid_actions}
            action_index = max(action_q_values, key=action_q_values.get) if maximize else min(action_q_values, key=action_q_values.get)
            return action_index

    def train(self, batch_size, gamma=0.99):
        if len(self.replay_buffer) < batch_size:
            print(f"Kickstarting the engine, waiting for {len(self.replay_buffer)} 100 entries...")
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def _state_to_tensor(self, state):
        hunter_x, hunter_y = state["hunter_pos"]
        pray_x, pray_y = state["pray_pos"]
        
        hunter_sees_pray = 1 if tuple(state["pray_pos"]) in state["hunter_vision"] else 0
        pray_sees_hunter = 1 if tuple(state["hunter_pos"]) in state["pray_vision"] else 0
        
        tensor = torch.tensor([
            hunter_x, hunter_y, pray_x, pray_y,
            abs(hunter_x - pray_x), abs(hunter_y - pray_y),
            hunter_sees_pray, pray_sees_hunter
        ], dtype=torch.float32).unsqueeze(0)
        return tensor


    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


class HunterModel(BaseModel):
    def __init__(self):
        super().__init__(input_dim=8, output_dim=5, actions=["UP", "DOWN", "LEFT", "RIGHT", "STAY"])

class PrayModel(BaseModel):
    def __init__(self):
        super().__init__(input_dim=8, output_dim=5, actions=["UP", "DOWN", "LEFT", "RIGHT", "STAY"])



def save_checkpoint(model, optimizer, epoch, epsilon, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'epsilon': epsilon
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(episodes, model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    epsilon = checkpoint['epsilon']
    print(f"Checkpoint loaded from {filename}, starting at epoch {epoch+1}")
    return epoch, epsilon

def train_game(episodes, grid_size, turns, batch_size, target_update_interval=10):
    hunter_model = HunterModel()
    pray_model = PrayModel()

    episode_rewards_hunter = []
    episode_rewards_pray = []

    #start_episode, hunter_model.epsilon = load_checkpoint(hunter_model.model, hunter_model.optimizer, "hunter_checkpoint.pth")
    #start_episode, pray_model.epsilon = load_checkpoint(pray_model.model, pray_model.optimizer, "pray_checkpoint.pth")

    for episode in range(episodes):
        if episode % target_update_interval == 0:
            hunter_model.target_model.load_state_dict(hunter_model.model.state_dict())
            pray_model.target_model.load_state_dict(pray_model.model.state_dict())

        game = Game(grid_size, turns)
        done = False

        total_reward_hunter = 0
        total_reward_pray = 0

        for turn in range(turns):
            if done:
                break

            state = game.get_state()
            hunter_action = hunter_model.predict(state, game.walls)
            pray_action = pray_model.predict(state, game.walls)

            next_state, reward_hunter, reward_pray, done = game.step(hunter_action, pray_action, turn)

            total_reward_hunter += reward_hunter
            total_reward_pray += reward_pray

            hunter_model.remember(state, hunter_model.actions.index(hunter_action), reward_hunter, next_state, done)
            pray_model.remember(state, pray_model.actions.index(pray_action), reward_pray, next_state, done)

            hunter_model.train(batch_size=batch_size)
            pray_model.train(batch_size=batch_size)

            os.system("cls" if os.name == "nt" else "clear")

            if (episode + 1) % 10 == 0: #False: 
                game.render_field()
            else:
                print(f"--- Episode {episode + 1} ---\n")
                print(f"Turn {turn + 1}")

        episode_rewards_hunter.append(total_reward_hunter)
        episode_rewards_pray.append(total_reward_pray)

        print("Game Over!\n")

        if (episode + 1) % 100 == 0:
            save_checkpoint(hunter_model.model, hunter_model.optimizer, episode, hunter_model.epsilon, f"hunter_checkpoint{episode+1}.pth")
            save_checkpoint(pray_model.model, pray_model.optimizer, episode, pray_model.epsilon, f"pray_checkpoint{episode+1}.pth")
    
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    window_size = 50

    plt.figure(figsize=(10, 6))

    reversed_rewards_pray = episode_rewards_pray[::-1]
    plt.plot(episode_rewards_hunter, color='green', alpha=0.5, label='Hunter Reward')
    plt.plot(episode_rewards_pray, color='blue', alpha=0.5, label='Pray Reward')

    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


train_game(episodes=1000, grid_size=20, turns=400, batch_size=32)
