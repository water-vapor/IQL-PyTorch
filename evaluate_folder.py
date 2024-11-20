from evaluate import main
import argparse
import os

def eval_folder(args):
    print(args)
    folder_name = f'{args.env}_{args.act}_{args.obs}'
    folder_path = os.path.join(args.log_dir, folder_name)
    subpath = os.listdir(folder_path)
    assert len(subpath) == 1
    ckpt_folder = os.path.join(folder_path, subpath[0])
    args.eval = True
    args.video = False
    args.savearr = False
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
    best_steps_mean = -1e10
    best_steps_std = -1e10
    for i, (mean, std, steps_mean, steps_std) in enumerate(results):
        if mean > best_mean:
            best_idx = i
            best_mean = mean
            best_std = std
            best_steps_mean = steps_mean
            best_steps_std = steps_std
    print(f'best ckpt: {ckpt_list[best_idx]}, mean: {best_mean}, std: {best_std}, steps_mean: {best_steps_mean}, steps_std: {best_steps_std}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='antmaze-umaze-v2')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--multistart', action='store_true')
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--obs', type=str, default='cartesian')
    parser.add_argument('--saveresults', action='store_true')
    args = parser.parse_args()
    eval_folder(args)