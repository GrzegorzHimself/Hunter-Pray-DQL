import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

from HnP_Train import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- REPLAY BUFFER ------------------- #
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        actions = np.array(actions).flatten()
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
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, lr=0.001, sequence_length=1):
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
        self.sequence_length = sequence_length
        self.state_buffer = []
        
    def reset_buffer(self):
        self.state_buffer = []
        
    def predict(self, state, hidden=None):
        self.state_buffer.append(state)
        if len(self.state_buffer) < self.sequence_length:
            padded_sequence = self.state_buffer + [state] * (self.sequence_length - len(self.state_buffer))
        else:
            padded_sequence = self.state_buffer[-self.sequence_length:]
            
        sequence_tensor = torch.tensor(np.array(padded_sequence), dtype=torch.float32).unsqueeze(0).to(device)
        
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
            return action, hidden
        else:
            with torch.no_grad():
                q_values, hidden = self.model(sequence_tensor, hidden)
                action = torch.argmax(q_values, dim=1).item()
            return action, hidden
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.unsqueeze(1)       # [batch, seq_len=1, input_dim]
        next_states = next_states.unsqueeze(1)
        q_values, _ = self.model(states)
        q_values = q_values.gather(1, actions.view(-1, 1)).squeeze(-1)
        with torch.no_grad():
            next_q_values, _ = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()