import os
os.environ['LD_LIBRARY_PATH'] = ':/home/vapor/miniconda3/lib/:/home/vapor/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/nvidia'
from pathlib import Path

import gym
import d4rl
import numpy as np
import torch
import imageio
import argparse

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy
from main import ReLULikeRescaledFittedExponentialActivationg2TTT, TanhLikeRescaledFittedExponentialActivationg2TTT

def main(args):
    if args.act == 'relu':
        act = torch.nn.ReLU
    elif args.act == 'tanh':
        act = torch.nn.Tanh
    elif args.act == 'fittedrelu':
        act = ReLULikeRescaledFittedExponentialActivationg2TTT
    elif args.act == 'fittedtanh':
        act = TanhLikeRescaledFittedExponentialActivationg2TTT
    else:
        raise NotImplementedError
    
    env_name = args.env
    pt_path = args.ckpt

    trained_model = torch.load(pt_path)
    env = gym.make(env_name, non_zero_reset=args.multistart)

    dataset = d4rl.qlearning_dataset(env)

    # for ant maze
    dataset['rewards'] -= 1.
    for k, v in dataset.items():
        dataset[k] = torchify(v)

    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    print(f'obs_dim: {obs_dim}, act_dim: {act_dim}')

    # defined in parameters
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=256, n_hidden=2, act=act)

    def eval_policy(n_steps=1000, n_episodes=100):
        eval_returns = np.array([evaluate_policy(env, policy, n_steps) \
                                    for _ in range(n_episodes)])
        normalized_returns = d4rl.get_normalized_score(env_name, eval_returns) * 100.0
        print({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=256, n_hidden=2),
        vf=ValueFunction(obs_dim, hidden_dim=256, n_hidden=2),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=3e-4),
        max_steps=10**6,
        tau=0.7,
        beta=3.0,
        alpha=0.005,
        discount=0.99,
    )

    iql.load_state_dict(trained_model)
    if args.eval:
        eval_policy()

    if args.video:
        env = gym.make(env_name, non_zero_reset=args.videomultistart)
        print(env, type(env))
        obs = env.reset()
        print(obs)
        fps = 30
        for run in range(5):
            writer = imageio.get_writer(f'{env_name}_good_run_{run}.mp4', format='FFMPEG', mode='I', fps=fps, quality=10)
            for _ in range(1000):
                action = iql.policy.act(torch.tensor(obs, dtype=torch.float32).cuda().reshape(1, -1)).cpu().numpy()[0]
                obs, reward, done, info = env.step(action)
                writer.append_data(env.render("rgb_array", width=1024, height=1024))
                if done:
                    obs = env.reset()
                    break
            writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='antmaze-umaze-v2')
    parser.add_argument('--ckpt', type=str, default='logs/antmaze-umaze-v2/08-31-23_13.03.56_wjkn/final.pt')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--videomultistart', action='store_true')
    parser.add_argument('--multistart', action='store_true')
    parser.add_argument('--act', type=str, default='relu')
    args = parser.parse_args()
    main(args)