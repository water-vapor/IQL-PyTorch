import csv
from datetime import datetime
import json
from pathlib import Path
import random
import string
import sys

import numpy as np
import torch
import torch.nn as nn


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HistogramRecorder:
    def __init__(self, min=-20, max=20):
        super().__init__()
        self.min = min
        self.max = max
        self.hist = torch.zeros(10*(self.max-self.min), dtype=torch.float32)

    def record(self, x):
        x = x.clone().detach().cpu()
        _hist = torch.histogram(x, bins=10*(self.max-self.min), range=(self.min, self.max)).hist
        self.hist += _hist

    def print(self):
        print(self.hist)

    def reset(self):
        self.hist = torch.zeros(10*(self.max-self.min), dtype=torch.float32)

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
    
class ReLU(HistogramRecorder, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.record(x)
        return torch.relu(x)
    
class Tanh(HistogramRecorder, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.record(x)
        return torch.tanh(x)
    
class FittedExponentialActivationg2TTT(FittedFunction):
    def __init__(self, clamp=False):
        def fn(x):
            return 0.880456*torch.exp(-7.31405*x+0.499283)-1.45059
        super().__init__(fn, -0.6, 0.0, clamp=clamp)


class FittedExponentialLinearActivation1010(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(
                x < -0.23,
                -120.202*x-11.7279,
                torch.where(
                    x < 0.4,
                    1.5889*torch.exp(0.567714-7.55103*x),
                    torch.zeros_like(x, device=x.device)
                )
            )
    
class FittedExponentialLinearActivation1010Eval(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(
            x < -0.4,
            torch.zeros_like(x, device=x.device),
            torch.where(
                x < -0.23,
                -120.202*x-11.7279,
                torch.where(
                    x < 0.4,
                    1.5889*torch.exp(0.567714-7.55103*x),
                    torch.zeros_like(x, device=x.device)
                )
            )
        )
    
class FittedELUActivationV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = FittedExponentialLinearActivation1010()

    def forward(self, x):
        return 0.5*(self.act(-0.02*x)) - 1
        # return (0.83*(self.act(-0.01*x-0.23)) - 13)/5

        # this is good
        # return torch.where(x<0, (0.83*(self.act(-0.01*x-0.23)) - 13)/5, (0.83*(self.act(-0.01*x-0.23)) - 13))
    

class FittedELUActivationV2Eval(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = FittedExponentialLinearActivation1010Eval()

    def forward(self, x):
        return 0.5*(self.act(-0.02*x)) - 1
        # return 0.83*(self.act(-0.01*x-0.23)) - 13

class ReLULikeRescaledFittedExponentialActivationg2TTT(HistogramRecorder, nn.Module):
    def __init__(self, clamp=False):
        super().__init__()
        self.clamp = clamp
        self.act = FittedExponentialActivationg2TTT(clamp=clamp)

    def forward(self, x):
        self.record(x)
        return 1/60*(self.act(-0.1*x)+1.45059)

class TanhLikeRescaledFittedExponentialActivationg2TTT(HistogramRecorder, nn.Module):
    def __init__(self, clamp=False):
        super().__init__()
        self.clamp = clamp
        self.act = FittedExponentialActivationg2TTT(clamp=clamp)

    def forward(self, x):
        self.record(x)
        return torch.sign(x)*(-1/60*self.act(torch.abs(x)-0.5) + 0.9125612803489692)

class ELULikeRescaledFittedExponentialActivationg2TTT(HistogramRecorder, nn.Module):
    def __init__(self, clamp=False):
        super().__init__()
        self.clamp = clamp
        self.act = FittedExponentialActivationg2TTT(clamp=clamp)

    def forward(self, x):
        self.record(x)
        # return torch.where(x >= 0, x, 
        #                    torch.where(
        #                        x <= -3.75,
        #                        torch.zeros_like(x, device=x.device),
        #                          0.01826361366142253*self.act(-0.13319443393572447*x-0.5)
        #                    ))
        return torch.where(x >= 0, x, 
                           torch.where(
                               x <= -3.75,
                               torch.zeros_like(x, device=x.device) - 1,
                                 0.01826361366142253*self.act(-0.13319443393572447*x-0.5) - 1
                           ))
    

class StandardSigmoidFn(FittedFunction):
    def __init__(self, clamp=False):
        def fn(x):
            return 1/(1+torch.exp(-x))
        super().__init__(fn, -1., 1., clamp=clamp)
    

class FittedSigmoid2024JanFn(FittedFunction):
    def __init__(self, clamp=False):
        def fn(x):
            return 1/(1+torch.exp(-12.55*(x-0.03)))
        super().__init__(fn, -1., 1., clamp=clamp)
    

class FittedSigmoid2024Jan(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = FittedSigmoid2024JanFn(True)

    def forward(self, x):
        return self.act(0.5*x)#4 -0.407
    

class ObservationConverter:
    def __init__(self, target, max_distance, obs_type):
        self.target = target
        self.max_distance = max_distance
        self.obs_type = obs_type
        if self.obs_type == 'polar':
            self.obs_size = 3
        elif self.obs_type == 'polar_prod':
            self.obs_size = 5
        elif self.obs_type == 'polar_prod2':
            self.obs_size = 11
        else:
            self.obs_size = 2

    def __call__(self, obs):
        xy = obs[:, :2]
        relative_xy = self.target - xy
        residue = obs[:, 2:]
        if self.obs_type == 'cartesian':
            return obs
        elif self.obs_type == 'polar':
            magnitude = torch.norm(relative_xy, dim=1, keepdim=True)
            unit_vec = relative_xy / magnitude
            normalized_magnitude = 1 - magnitude / self.max_distance
            return torch.cat([unit_vec, normalized_magnitude, residue], dim=1)
        elif self.obs_type == 'polar_prod':
            magnitude = torch.norm(relative_xy, dim=1, keepdim=True)
            unit_vec = relative_xy / magnitude
            unit_vec_mag_prod = unit_vec * magnitude
            normalized_magnitude = 1 - magnitude / self.max_distance
            return torch.cat([unit_vec, normalized_magnitude, unit_vec_mag_prod, residue], dim=1)
        elif self.obs_type == 'cartesian_relative':
            return torch.cat([relative_xy, residue], dim=1)
        elif self.obs_type == 'polar2':
            angle = torch.atan2(relative_xy[:, 1], relative_xy[:, 0]).unsqueeze(1)
            magnitude = torch.norm(relative_xy, dim=1, keepdim=True)
            normalized_magnitude = 1 - magnitude / self.max_distance
            return torch.cat([angle, normalized_magnitude, residue], dim=1)
        else:
            raise NotImplementedError
    

def get_obs_converter(env_name, obs_type, device=DEFAULT_DEVICE):
    if env_name.startswith('antmaze-umaze'):
        target = [0., 8.]
        max_distance = 15.
    elif env_name.startswith('antmaze-medium'):
        target = [20., 20.]
        max_distance = 30.
    elif env_name.startswith('antmaze-large'):
        target = [32., 24.]
        max_distance = 41.
    else:
        raise NotImplementedError
    obs_converter = ObservationConverter(torch.tensor(target).to(device), max_distance, obs_type)
    return obs_converter


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU(), output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i == n_dims - 3:
            layers.append(activation)
        else:
            layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation)
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x



def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices] for k, v in dataset.items()}


def evaluate_policy(env, policy, max_episode_steps, deterministic=True, obs_converter=None, return_steps=False):
    obs = env.reset()
    total_reward = 0.
    for s in range(max_episode_steps):
        with torch.no_grad():
            obs = torchify(obs).reshape(1, -1)
            if obs_converter is not None:
                obs = obs_converter(obs)
            action = policy.act(obs, deterministic=deterministic).cpu().numpy()[0]
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    if return_steps:
        if s == max_episode_steps - 1:
            return total_reward, -1
        else:
            return total_reward, s
    else:
        return total_reward

def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()