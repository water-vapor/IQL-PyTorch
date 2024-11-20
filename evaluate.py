import gym
import d4rl
import numpy as np
import torch
import imageio
import argparse

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import *

def main(args):
    if args.act == 'relu':
        act = torch.nn.ReLU()
    elif args.act == 'tanh':
        act = torch.nn.Tanh()
    elif args.act == 'fittedrelu':
        act = ReLULikeRescaledFittedExponentialActivationg2TTT()
    elif args.act == 'fittedtanh':
        act = TanhLikeRescaledFittedExponentialActivationg2TTT()
    elif args.act == 'fittedelu':
        act = ELULikeRescaledFittedExponentialActivationg2TTT()
    elif args.act == 'fittedeluv2':
        act = FittedELUActivationV2Eval()
    elif args.act == 'fittedsig':
        act = FittedSigmoid2024Jan()
    else:
        raise NotImplementedError
    
    obs_converter = get_obs_converter(args.env, args.obs)
    
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
    obs_dim = (obs_dim - 2 + obs_converter.obs_size)
    act_dim = dataset['actions'].shape[1]
    print(f'obs_dim: {obs_dim}, act_dim: {act_dim}')

    # defined in parameters
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=256, n_hidden=2, act=act)

    def eval_policy(n_steps=1000, n_episodes=100, return_results=False):
        results = [evaluate_policy(env, policy, n_steps, obs_converter=obs_converter, return_steps=True) \
                                    for _ in range(n_episodes)]
        eval_returns = np.array([r[0] for r in results])
        steps = np.array([r[1] for r in results if r[1] != -1])
        steps_mean = steps.mean()
        steps_std = steps.std()
        normalized_returns = d4rl.get_normalized_score(env_name, eval_returns) * 100.0
        print(f'mean: {normalized_returns.mean()}, std: {normalized_returns.std()}, steps: {steps_mean}, steps_std: {steps_std}')
        if return_results:
            return steps
        else:
            return normalized_returns.mean(), normalized_returns.std(), steps_mean, steps_std

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
        obs_converter=obs_converter
    )

    iql.load_state_dict(trained_model)
    if args.eval:
        if hasattr(args, 'saveresults') and args.saveresults:
            print('saving results')
            result = eval_policy(n_steps=2000, return_results=True)
            np.save(f'{env_name}_{args.act}_{args.saveresultspostfix}_eval_returns.npy', result)
        else:
            result = eval_policy()

    if args.video:
        env = gym.make(env_name, non_zero_reset=args.videomultistart)
        print(env, type(env))
        obs = env.reset()
        fps = 30
        for run in range(5):
            writer = imageio.get_writer(f'{env_name}_good_run_{run}.mp4', format='FFMPEG', mode='I', fps=fps, quality=10)
            for _ in range(1000):
                obs_torch = torch.tensor(obs, dtype=torch.float32).cuda().reshape(1, -1)
                obs_torch = obs_converter(obs_torch)
                action = iql.policy.act(obs_torch).cpu().numpy()[0]
                obs, reward, done, info = env.step(action)
                writer.append_data(env.render("rgb_array", width=1024, height=1024))
                if done:
                    obs = env.reset()
                    break
            writer.close()

    # separated because some servers can't render correctly
    if args.savearr:
        env = gym.make(env_name, non_zero_reset=args.multistart)
        obs = env.reset()
        for run in range(5):
            observations = []
            actions = []
            for _ in range(1000):
                obs_torch = torch.tensor(obs, dtype=torch.float32).cuda().reshape(1, -1)
                obs_torch = obs_converter(obs_torch)
                action = iql.policy.act(obs_torch).cpu().numpy()[0]
                observations.append(obs)
                actions.append(action)
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
                    break
            np.save(f'{env_name}_good_run_{run}_obs.npy', np.array(observations))
            np.save(f'{env_name}_good_run_{run}_act.npy', np.array(actions))

    if args.eval:
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='antmaze-umaze-v2')
    parser.add_argument('--ckpt', type=str, default='logs/antmaze-umaze-v2/08-31-23_13.03.56_wjkn/final.pt')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--videomultistart', action='store_true')
    parser.add_argument('--multistart', action='store_true')
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--savearr', action='store_true')
    parser.add_argument('--saveresults', action='store_true')
    parser.add_argument('--saveresultspostfix', type=str, default='')
    parser.add_argument('--obs', type=str, default='cartesian')
    args = parser.parse_args()
    print(main(args))