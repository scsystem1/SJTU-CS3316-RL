import numpy as np
import torch
import torch.nn as nn
import collections
import random
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Replay_Buffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen = buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return [np.array(states), actions, rewards, np.array(next_states), dones]

    def size(self):
        return len(self.buffer)

class DuelingCnnQNet(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(DuelingCnnQNet, self).__init__()

        self.input_shape = input_shape
        self.output_dim = output_dim

        self.share_feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, x):
        x = self.share_feature(x)
        x = x.view(x.size(0), -1)
        value = self.value(x)
        advantage = self.advantage(x)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


    def feature_size(self):
        with torch.no_grad():
            return self.share_feature(torch.zeros(1, *self.input_shape)).view(1, -1).shape[1]


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, output_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(output_dim).to(device))

    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

def eps_decay(eps_start, eps_end, eps_decay_steps, frame_idx):
    return eps_end + (eps_start - eps_end) * np.exp(-frame_idx / eps_decay_steps)

def evaluate(env, agent, total_frames=50000):
    agent.epsilon = 0
    return_list = []
    state, _ = env.reset()
    done = False
    sum_reward = 0
    for _ in range(total_frames):
        if done:
            return_list.append(sum_reward)
            state, _ = env.reset()
            done = False
            sum_reward = 0
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        sum_reward += reward
        state = next_state
    return return_list

def preprocess_env(env: gym.Env):
  env = AtariPreprocessing(
    env,
    frame_skip=1,
    screen_size=84,
    terminal_on_life_loss=False,
    grayscale_obs=True,
    grayscale_newaxis=False,
    scale_obs=True
  )
  env = FrameStack(env, num_stack=4)
  return env

def save_model(model, save_dir, filename='dueling_ddqn.pth'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to: {save_path}')