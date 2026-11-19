"""Run only missing UAV-count and communication-load experiments.
Place beside run_research_experiments.py in the project root.
Usage: python run_gap_experiments.py --gap uavs --updates 10
       python run_gap_experiments.py --gap traffic --updates 10
"""
import argparse
import sys
import run_research_experiments as base


def fixed_cases(mode, grid):
    common = dict(actor_lr=3e-5, critic_lr=1e-4, minibatch=256, tx=0.005)
    if mode == 'scalability':
        return [dict(common, uavs=k, iot=12) for k in (2, 3, 4)]
    if mode == 'traffic':
        return [dict(common, uavs=3, iot=12, tx=tx)
                for tx in (0.0005, 0.005, 0.01)]
    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser(description='Missing results experiments')
    parser.add_argument('--gap', choices=['uavs', 'traffic'], required=True)
    parser.add_argument('--updates', type=int, default=10)
    parser.add_argument('--rollout-steps', type=int, default=256)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--train-seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--output-dir', default='pls_mappo/research_gaps')
    args = parser.parse_args()
    if min(args.updates, args.rollout_steps, args.eval_episodes, len(args.train_seeds)) <= 0:
        parser.error('positive sizes required')
    mode = 'scalability' if args.gap == 'uavs' else 'traffic'
    # The Stage 8 runner already implements training, held-out evaluation,
    # checkpoint handling, and CSV saving. Override only its experiment cases.
    base.get_cases = fixed_cases
    output = args.output_dir + ('_fixed_iot12' if args.gap == 'uavs' else '_tx_demand')
    sys.argv = [sys.argv[0], '--mode', mode,
                '--updates', str(args.updates),
                '--rollout-steps', str(args.rollout_steps),
                '--eval-episodes', str(args.eval_episodes),
                '--train-seeds', *(str(s) for s in args.train_seeds),
                '--output-dir', output]
    print('Experiment:', args.gap, '| output root:', output, flush=True)
    base.main()


if __name__ == '__main__':
    main()
