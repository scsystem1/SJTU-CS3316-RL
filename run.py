import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
import torch
from tqdm import tqdm
import random
from utils import Replay_Buffer, eps_decay, evaluate, preprocess_env, save_model, RunningMeanStd
from dqn import DuelingDDQN
from ppo import PPO
import argparse
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gym.register_envs(ale_py)

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, required=True)
args = parser.parse_args()

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

if args.env_name == 'ALE/Pong-v5':
    env = gym.make(args.env_name)
    env = preprocess_env(env)
    input_shape = (env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2])
    output_dim = env.action_space.n
    l_r = 0.0001
    gamma = 0.99
    eps_start = 0.1
    eps_end = 0.01
    eps_decay_steps = 50000
    target_update_step = 1000
    buffer_size = 50000
    mini_size = 10000
    batch_size = 64
    num_frames = 2000000

    buffer = Replay_Buffer(buffer_size)
    agent = DuelingDDQN(input_shape, output_dim, l_r, eps_start, gamma, target_update_step)
    return_list = []
    loss_list = []
    state, _ = env.reset()
    done = False
    sum_reward = 0
    dones = 0
    with tqdm(total=num_frames, desc=f'Iteration') as pbar:
        for frame in range(num_frames):
            agent.epsilon = eps_decay(eps_start, eps_end, eps_decay_steps, frame)
            if done:
                return_list.append(sum_reward)
                state, _ = env.reset()
                done = False
                sum_reward = 0
                dones += 1
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            #state = np.transpose(state, (2, 0, 1))
            #next_state_prep = np.transpose(next_state, (2, 0, 1))
            buffer.add((state, action, reward, next_state, done))
            sum_reward += reward
            if buffer.size() > mini_size:
                experience_batch = buffer.sample(batch_size)
                loss = agent.update(experience_batch)
                loss_list.append(loss)
            state = next_state
            if dones % 10 == 0:
                if len(return_list) >= 10:
                    avg_return = np.mean(return_list[-10:])
                else:
                    avg_return = np.mean(return_list) if return_list else 0

                if len(loss_list) >= 50:
                    avg_loss = np.mean(loss_list[-50:])
                else:
                    avg_loss = np.mean(loss_list) if loss_list else 0

                pbar.set_postfix({
                    'frames': frame,
                    'avg_return': f'{avg_return:.2f}',
                    'avg_loss': f'{avg_loss:.4f}'
                })
            pbar.update(1)

    save_model(agent.q_net, './models', filename='pong_ddqn_final.pth')
    return_list = np.array(return_list)
    loss_list = np.array(loss_list)
    return_list_eval = np.array(evaluate(env, agent))
    best_performance = np.max(return_list_eval)
    print(f'Best Performance: {best_performance}')

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(return_list)
    plt.xlabel('Round')
    plt.ylabel('Return')
    plt.title('Training_Return: Dueling DDQN on Pong-v5')

    plt.subplot(2, 2, 2)
    plt.plot(np.convolve(return_list, np.ones(10) / 10, mode='valid'))
    plt.xlabel('Window')
    plt.ylabel('Average Return(10 episodes)')
    plt.title('Training_Return: Dueling DDQN on Pong-v5')

    plt.subplot(2, 2, 3)
    plt.plot(np.convolve(loss_list, np.ones(200) / 200, mode='valid'))
    plt.xlabel('Window')
    plt.ylabel('Average Loss(200 episodes)')
    plt.title('Training_Loss: Dueling DDQN on Pong-v5')

    plt.subplot(2, 2, 4)
    plt.plot(return_list_eval)
    plt.xlabel('Round')
    plt.ylabel('Return')
    plt.title('Evaluation_Return: Dueling DDQN on Pong-v5')

    plt.tight_layout()
    plt.savefig('Dueling_DDQN_on_Pong-v5.png')

    plt.show()

else:
    env = gym.make(args.env_name, render_mode="rgb_array")
    input_shape = env.observation_space.shape
    output_dim = env.action_space.shape[0]
    if args.env_name == 'Hopper-v4':
        entropy_coef = 1e-3
        actor_lr = 3e-4
        critic_lr = 3e-4
        gamma = 0.99
        lambda_ = 0.95
        max_clip = 0.2
        train_epoch = 5
        traj_length = 2048
        batch_size = 64
        max_grad_norm = 0.5
        num_episodes = 1500
    elif args.env_name == 'Ant-v4':
        entropy_coef = 1e-3
        actor_lr = 3e-4
        critic_lr = 3e-4
        gamma = 0.99
        lambda_ = 0.95
        max_clip = 0.2
        train_epoch = 10
        traj_length = 2048
        batch_size = 128
        max_grad_norm = 0.5
        num_episodes = 2000
    elif args.env_name == 'HalfCheetah-v4':
        entropy_coef = 1e-3
        actor_lr = 3e-4
        critic_lr = 3e-4
        gamma = 0.99
        lambda_ = 0.95
        max_clip = 0.2
        train_epoch = 10
        traj_length = 2048
        batch_size = 128
        max_grad_norm = 0.5
        num_episodes = 2000

    agent = PPO(input_shape[0], output_dim, actor_lr, critic_lr, gamma, lambda_, max_clip,
                train_epoch, entropy_coef, max_grad_norm)
    return_list = []
    state_rms = RunningMeanStd(input_shape[0])
    if args.env_name == 'Ant-v4' or args.env_name == 'HalfCheetah-v4':
        reward_rms = RunningMeanStd(1)

    for i in range(10):
        with tqdm(total=num_episodes // 10, desc=f'Iteration {i + 1}') as pbar:
            for ep in range(num_episodes // 10):
                trajectory = []
                raw_trajectory = []
                if args.env_name == 'Ant-v4' or args.env_name == 'HalfCheetah-v4':
                    reward_list = []
                sum_reward = 0
                steps = 0

                while steps < traj_length:
                    state, _ = env.reset()
                    done = False

                    while not done and steps < traj_length:
                        raw_trajectory.append(state)
                        norm_state = np.clip((state - state_rms.mean) / (np.sqrt(state_rms.var) + 1e-8), -5, 5)

                        action = agent.take_action(norm_state)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        norm_next_state = np.clip((next_state - state_rms.mean) / (np.sqrt(state_rms.var) + 1e-8), -5,
                                                  5)
                        done = terminated or truncated
                        if args.env_name == 'Ant-v4' or args.env_name == 'HalfCheetah-v4':
                            reward_list.append([reward])
                            reward_rms.update(np.array(reward_list))
                            scaled_reward = (reward - reward_rms.mean) / (np.sqrt(reward_rms.var) + 1e-8)
                        if args.env_name == 'Hopper-v4':
                            trajectory.append((norm_state, action, reward, norm_next_state, done))
                        else:
                            trajectory.append((norm_state, action, scaled_reward, norm_next_state, done))
                        sum_reward += reward
                        steps += 1
                        state = next_state

                    if done:
                        return_list.append(sum_reward)
                    sum_reward = 0

                batch = list(zip(*trajectory))
                agent.update(batch)
                state_rms.update(np.array(raw_trajectory))

                if (ep + 1) % 5 == 0:
                    avg_return = np.mean(return_list[-5:])
                    pbar.set_postfix({
                        'episode': i * (num_episodes // 10) + ep + 1,
                        'avg_return': f'{avg_return:.2f}'
                    })

                pbar.update(1)

    save_model(agent.actor, './models', filename=f'{args.env_name}_ppo_actor.pth')
    save_model(agent.critic, './models', filename=f'{args.env_name}_ppo_critic.pth')
    return_list = np.array(return_list)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(return_list)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'Training_Return: PPO on {args.env_name}')

    plt.subplot(1, 2, 2)
    plt.plot(np.convolve(return_list, np.ones(20) / 20, mode='valid'))
    plt.xlabel('Window')
    plt.ylabel('Average Return(20 episodes)')
    plt.title(f'Training_Return: PPO on {args.env_name}')

    plt.tight_layout()
    plt.savefig(f'PPO_on_{args.env_name}.png')

    plt.show()

    video_folder = "./videos"
    env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)
    state, _ = env.reset()
    done = False

    total_reward = 0
    while not done:
        norm_state = np.clip((state - state_rms.mean) / (np.sqrt(state_rms.var) + 1e-8), -5, 5)
        action = agent.take_action(norm_state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    print(f"Saved video to {Path(video_folder).resolve()}")