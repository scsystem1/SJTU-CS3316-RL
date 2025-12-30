import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, input_dim, output_dim, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99,
                 lam=0.95, clip_epsilon=0.2, epochs=10, entropy_coef=0.01, max_grad_norm = 0.5):
        self.actor = Actor(input_dim, output_dim).to(device)
        self.critic = Critic(input_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        mu, std = self.actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        return action.cpu().numpy()[0]

    def compute_gae(self, rewards, dones, values, next_values):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = []
        adv = 0.0
        for delta, done in zip(reversed(deltas), reversed(dones)):
            adv = delta + self.gamma * self.lam * adv * (1 - done)
            advantages.insert(0, adv)
        return torch.tensor(advantages, dtype=torch.float32)

    def update(self, batch):
        state = torch.tensor(batch[0], dtype=torch.float32).to(device)
        action = torch.tensor(batch[1], dtype=torch.float32).to(device)
        reward = torch.tensor(batch[2], dtype=torch.float32).view(-1, 1).to(device)
        next_state = torch.tensor(batch[3], dtype=torch.float32).to(device)
        done = torch.tensor(batch[4], dtype=torch.float32).view(-1, 1).to(device)

        value = self.critic(state)
        next_value = self.critic(next_state)
        advantage = self.compute_gae(reward, done, value, next_value).view(-1, 1).to(device)
        td_target = advantage + value
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-3).to(device)

        with torch.no_grad():
            mu_old, std_old = self.actor(state)
            dist_old = torch.distributions.Normal(mu_old, std_old)
            log_prob_old = dist_old.log_prob(action).sum(dim=-1, keepdim=True)

        for _ in range(self.epochs):
            mu, std = self.actor(state)
            dist = torch.distributions.Normal(mu, std)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)

            ratio = torch.exp(log_prob - log_prob_old)
            object1 = ratio * advantage
            object2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
            actor_loss = -torch.min(object1, object2).mean() - self.entropy_coef * entropy.mean()

            critic_loss = F.mse_loss(self.critic(state), td_target.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()