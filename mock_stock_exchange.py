import pandas as pd
import datetime
import os
import pickle as pk
from stock_price_model_tensorflow import ModelMaker
import tensorflow as tf
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Ensures output is in range [-1, 1]
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action = self.actor(state)
        return action

    def evaluate(self, state):
        value = self.critic(state)
        return value

class Trader:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.policy_old.act(state)

        memory.states.append(state)
        memory.actions.append(action)
        action = action.detach().cpu().numpy().flatten()
        return action

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device)).detach()

        for _ in range(self.K_epochs):
            logprobs = self.policy.act(old_states)
            state_values = self.policy.evaluate(old_states)
            dist_entropy = torch.mean(logprobs)
            advantages = rewards - state_values.detach()

            ratios = torch.exp(logprobs - old_actions)

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

# class to keep all the stock prices and give prices to the trader
class MarketEnv(gym.Env):
    def __init__(self, initial_cash=1000):
        super(MarketEnv, self).__init__()
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        '''
        ModelMaker containing input_n_output with dates in format of "%Y-%m-%d %H-%M-%S" and corresponding ticker
        creates model as well
        '''
        self.ModelMaker = ModelMaker(load_data=True)

        self.model = self.ModelMaker.train_model(train_model=False, model_name="model_temp3", epochs=3)
        self.output = self.ModelMaker.output
        self.pred_output = self.model.predict([self.ModelMaker.past, self.ModelMaker.quarter, self.ModelMaker.yearly, self.ModelMaker.info, self.ModelMaker.headlines])
        prices = np.concatenate((self.ModelMaker.dates, self.ModelMaker.tickers, self.ModelMaker.last_days, self.pred_output), axis=1)
        # with self.output: Episode 190      Average Reward: 1120981.97
        self.prices = {}
        for day in prices:
            if day[1] not in self.prices.keys():
                self.prices[day[1]] = [day]
            else: self.prices[day[1]].append(day)
        for ticker, price in self.prices.items():
            self.prices[ticker] = np.reshape(np.concatenate(price), (len(price), np.shape(price)[1]))
        self.initial_cash = initial_cash
        self.tickers = list(self.prices.keys())
        self.num_stocks = len(self.tickers)
        self.current_prices = np.zeros((self.num_stocks, np.shape(price)[1]-2))
        
        # Continuous action space where each action specifies quantity to buy or sell
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(np.shape(price)[1] - 2 + 3,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.cash = [self.initial_cash]
        self.stocks = np.zeros(self.num_stocks)
        self.update_current_prices()
        return self._get_observation()

    def update_current_prices(self):
        for i, ticker in enumerate(self.tickers):
            self.current_prices[i] = self.prices[ticker][self.current_step, 2:10] if self.current_step < len(self.prices[ticker]) else self.prices[ticker][-1, 2:10]

    def _get_observation(self):
        return np.hstack((np.tile(np.concatenate((self.cash, self.stocks)), (self.current_prices.shape[0], 1)), self.current_prices))

    def step(self, actions):
        self.update_current_prices()
        for i in range(self.num_stocks):
            action = actions[i]
            if action > 0:  # Buy
                quantity = min(action * self.cash / np.mean(self.current_prices[i][0:4]), self.cash // np.mean(self.current_prices[i][0:4]))
                self.stocks[i] += quantity
                self.cash -= quantity * np.mean(self.current_prices[i][0:4])
            elif action < 0:  # Sell
                quantity = min(-action * self.stocks[i], self.stocks[i])
                self.stocks[i] -= quantity
                self.cash += quantity * np.mean(self.current_prices[i][0:4])

        self.current_step += 1
        done = self.current_step >= len(self.prices[self.tickers[0]])
        self.update_current_prices()
        only_prices = np.array([[np.mean(i[:4])] for i in self.current_prices])
        reward = self.cash + np.sum(self.stocks * only_prices)

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        current_prices = self.current_prices
        print(f'Step: {self.current_step}, Cash: {self.cash}, Stocks: {self.stocks}, Prices: {current_prices}')

def train():
    env = MarketEnv(initial_cash=1000)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo_agent = Trader(state_dim, action_dim)
    memory = Memory()

    max_episodes = 1000
    max_timesteps = len(next(iter(env.prices.values())))
    update_timestep = 2000
    log_interval = 10

    timestep = 0
    rewards = []

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        for t in range(max_timesteps):
            timestep += 1
            actions = []
            for i, ticker in enumerate(env.tickers):
                actions.append(ppo_agent.select_action(state[i], memory))
            
            state, reward, done, _ = env.step(actions)
            for i in range(env.num_stocks):
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

