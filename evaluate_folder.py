from evaluate import main
import argparse
import os

def eval_folder(args):
    print(args)
    args.eval = True
    args.video = False
    args.savearr = False
    ckpt_folder = args.ckpt_folder
    ckpt_list = os.listdir(ckpt_folder)
    ckpt_list = [ckpt for ckpt in ckpt_list if ckpt.endswith('.pt')]
    ckpt_list.sort()
    results = []
    for ckpt in ckpt_list:
        ckpt_path = os.path.join(ckpt_folder, ckpt)
        args.ckpt = ckpt_path
        results.append(main(args))

    best_idx = -1
    best_mean = -1e10
    best_std = -1e10
    for i, (mean, std) in enumerate(results):
        if mean > best_mean:
            best_idx = i
            best_mean = mean
            best_std = std
    print(f'best ckpt: {ckpt_list[best_idx]}, mean: {best_mean}, std: {best_std}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='antmaze-umaze-v2')
    parser.add_argument('--ckpt_folder', type=str, default='logs/antmaze-umaze-v2_relu/09-08-23_00.15.18_pgsf')
    parser.add_argument('--multistart', action='store_true')
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--polar', action='store_true')
    args = parser.parse_args()
    eval_folder(args)