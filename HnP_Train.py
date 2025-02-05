import random
import numpy as np
import matplotlib.pyplot as plt

from HnP_Enviroment import *
from HnP_RNN import *


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