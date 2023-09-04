from pathlib import Path

import gym
import d4rl
import numpy as np
import torch
from torch import nn
from tqdm import trange

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy

class FittedFunction(nn.Module):
    def __init__(self, fn, range_min, range_max, clamp=False):
        super().__init__()
        self.fn = fn
        self.range_min = range_min
        self.range_max = range_max
        self.clamp = clamp

    def forward(self, x):
        if self.clamp:
            x = torch.clamp(x, self.range_min, self.range_max)
        return self.fn(x)
    
class FittedExponentialActivationg2TTT(FittedFunction):
    def __init__(self, clamp=False):
        def fn(x):
            return 0.880456*torch.exp(-7.31405*x+0.499283)-1.45059
        super().__init__(fn, -0.6, 0.0, clamp=clamp)

class ReLULikeRescaledFittedExponentialActivationg2TTT(nn.Module):
    def __init__(self, clamp=False):
        super().__init__()
        self.clamp = clamp
        self.act = FittedExponentialActivationg2TTT(clamp=clamp)

    def forward(self, x):
        return 1/60*(self.act(-0.1*x)+1.45059)

class TanhLikeRescaledFittedExponentialActivationg2TTT(nn.Module):
    def __init__(self, clamp=False):
        super().__init__()
        self.clamp = clamp
        self.act = FittedExponentialActivationg2TTT(clamp=clamp)

    def forward(self, x):
        return torch.sign(x)*(-1/60*self.act(torch.abs(x)-0.5) + 0.9125612803489692)


def get_env_and_dataset(log, env_name, max_episode_steps):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        log(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset


def main(args):
    torch.set_num_threads(1)
    log = Log(Path(args.log_dir)/f'{args.env_name}_{args.act}', vars(args))
    log(f'Log dir: {log.dir}')

    env, dataset = get_env_and_dataset(log, args.env_name, args.max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)

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

    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden, act=act)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden, act=act)
    def eval_policy():
        eval_returns = np.array([evaluate_policy(env, policy, args.max_episode_steps) \
                                 for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )

    for step in trange(args.n_steps):
        iql.update(**sample_batch(dataset, args.batch_size))
        if (step+1) % args.eval_period == 0:
            eval_policy()

    torch.save(iql.state_dict(), log.dir/'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument('--act', type=str, default='relu')
    main(parser.parse_args())