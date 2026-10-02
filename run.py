from __future__ import print_function

import argparse
import yaml
import os
import sys
from time import time

def main():
    start_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--max_oracle_calls', type=int, default=1000)
    parser.add_argument('--freq_log', type=int, default=10)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])
    parser.add_argument('--max_obj', nargs="*", default=[])
    parser.add_argument('--min_obj', nargs="*", default=[])
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--oracle', type=str, default='boltz', choices=["boltz", "docking"])
    args = parser.parse_args()

    if len(args.max_obj) + len(args.min_obj) == 0:
        parser.error("specify at least one objective via --max_obj or --min_obj")

    print(args)

    path_main = os.path.join(os.path.dirname(os.path.realpath(__file__)), "main", "molleo_multi_pareto")
    sys.path.append(path_main)

    from main.molleo_multi_pareto.run import GB_GA_Optimizer as Optimizer

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print(f'Maximizing oracle function: {args.max_obj}')
    print(f'Minimizing oracle function: {args.min_obj}')

    try:
        config_default = yaml.safe_load(open(args.config_default))
    except:
        config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

    directory_path = os.path.join(args.output_dir, args.run_name)
    os.makedirs(directory_path, exist_ok=True)
    args.output_dir = directory_path

    print(config_default)

    for seed in args.seed:
        optimizer = Optimizer(args=args, config=config_default)
        print('seed', seed)
        optimizer.optimize(seed=seed)

    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))


if __name__ == "__main__":
    main()
