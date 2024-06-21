import gym
from gym import spaces
import numpy as np

class StockMarketEnv(gym.Env):
    def __init__(self, prices, initial_cash=1000):
        super(StockMarketEnv, self).__init__()
        self.prices = prices
        self.initial_cash = initial_cash
        self.action_space = spaces.Discrete(3)  # [0: hold, 1: buy, 2: sell]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.stocks = 0
        return self._get_observation()

    def _get_observation(self):
        if self.current_step >= len(self.prices):
            current_price = self.prices[-1]
        else:
            current_price = self.prices[self.current_step]
        return np.array([self.cash, self.stocks, current_price])

    def step(self, action):
        current_price = self.prices[self.current_step]
        if action == 1 and self.cash >= current_price:  # Buy
            self.stocks += 1
            self.cash -= current_price
        elif action == 2 and self.stocks > 0:  # Sell
            self.stocks -= 1
            self.cash += current_price

        self.current_step += 1
        done = self.current_step >= len(self.prices)
        if done:
            reward = self.cash + self.stocks * self.prices[-1]
        else:
            reward = self.cash + self.stocks * self.prices[self.current_step]

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        current_price = self.prices[self.current_step-1] if self.current_step > 0 else self.prices[0]
        print(f'Step: {self.current_step}, Cash: {self.cash}, Stocks: {self.stocks}, Price: {current_price}')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_logits = self.action_head(x)
        state_value = self.value_head(x)
        return action_logits, state_value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = PPO(state_dim, action_dim)
        self.policy_old = PPO(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_logits, _ = self.policy_old(state)
        action_prob = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_prob)
        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action.item()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(memory.states).detach())
        old_actions = torch.squeeze(torch.stack(memory.actions).detach())
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs).detach())

        for _ in range(self.K_epochs):
            logprobs, state_values = self.policy(old_states)
            dist = Categorical(torch.softmax(logprobs, dim=-1))
            new_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            state_values = torch.squeeze(state_values)

            ratios = torch.exp(new_logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


import matplotlib.pyplot as plt

def train():
    prices = [100, 105, 102, 108, 107, 110, 113, 115, 112, 116]  # Extended sample stock prices
    env = StockMarketEnv(prices)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPOAgent(state_dim, action_dim)
    memory = Memory()

    max_episodes = 1000
    max_timesteps = len(prices)
    update_timestep = 2000
    log_interval = 10

    timestep = 0
    rewards = []

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        for t in range(max_timesteps):
            timestep += 1

            action = ppo_agent.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward

            if timestep % update_timestep == 0:
                ppo_agent.update(memory)
                memory.clear_memory()
                timestep = 0

            if done:
                break

        rewards.append(episode_reward)
        if episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            print(f'Episode {episode} \t Average Reward: {avg_reward:.2f}')

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

if __name__ == '__main__':
    train()


