import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import DuelingCnnQNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDDQN:
    def __init__(self, input_shape, output_dim, learning_rate=0.0001, epsilon=0.1, gamma=0.99, target_update_step=1000):
        self.action_dim = output_dim
        self.q_net = DuelingCnnQNet(input_shape, output_dim).to(device)
        self.target_net = DuelingCnnQNet(input_shape, output_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate, amsgrad=True)
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_update_step = target_update_step
        self.count = 0

    def take_action(self, state):
        #state = np.transpose(state, (2, 0, 1))
        if random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, batch):
        states = torch.tensor(batch[0], dtype=torch.float32).to(device)
        actions = torch.tensor(batch[1], dtype=torch.int64).view(-1, 1).to(device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).view(-1, 1).to(device)
        next_states = torch.tensor(batch[3], dtype=torch.float32).to(device)
        dones = torch.tensor(batch[4], dtype=torch.float32).view(-1, 1).to(device)

        q_cal = self.q_net(states).gather(1, actions)
        next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).detach()
        q_target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = F.mse_loss(q_cal, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        if self.count % self.target_update_step == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()