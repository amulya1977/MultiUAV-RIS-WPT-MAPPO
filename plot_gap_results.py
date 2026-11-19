"""Plot the three missing results from actual files generated in VS Code.
Usage: python plot_gap_results.py --what lr
       python plot_gap_results.py --what uavs
       python plot_gap_results.py --what traffic
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path('pls_mappo')
OUT = ROOT / 'research_graphs'
OUT.mkdir(parents=True, exist_ok=True)


def plot_lr():
    # Only the 10-update Stage 8 sensitivity runs at minibatch 256.
    paths = sorted(ROOT.glob('**/sensitivity/*/training_metrics.json'))
    found = {}
    for path in paths:
        cfg_path = path.parent / 'run_config.json'
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        if (int(cfg.get('minibatch_size', -1)) != 256 or
                int(cfg.get('num_uavs', -1)) != 3 or
                int(cfg.get('num_iot', -1)) != 12 or
                int(cfg.get('seed', -1)) != 42 or
                int(cfg.get('updates', -1)) != 10):
            continue
        a, c = float(cfg['actor_lr']), float(cfg['critic_lr'])
        label = None
        if abs(a - 3e-5) < 1e-10 and abs(c - 1e-4) < 1e-10:
            label = 'Reference: actor 0.00003, critic 0.0001'
        elif a == c and a in (0.01, 0.03, 0.05):
            label = f'Actor/Critic LR = {a:g}'
        if label is not None:
            found[label] = (path, json.loads(path.read_text(encoding='utf-8')))
    if len(found) != 4:
        print('Found:', *found, sep='\n  ')
        raise RuntimeError('Need four matching 10-update sensitivity histories at batch 256. '
                           'Locate your research_screen_fixed/sensitivity folders.')
    plt.figure(figsize=(10, 5))
    for label, (path, logs) in sorted(found.items()):
        x = [r['update'] for r in logs]
        y = [r['mean_reward'] for r in logs]
        plt.plot(x, y, marker='.', label=label)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('PPO Training Update (not episode)')
    plt.ylabel('Mean Signed Training Reward')
    plt.title('Four Learning-Rate Configurations: Reward vs PPO Update')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    dest = OUT / '22_four_learning_rates_reward_vs_update.png'
    plt.savefig(dest, dpi=300)
    plt.close()
    print('Saved:', dest)
    for name, (path, _) in found.items():
        print(name, '<-', path)


def find_summary(mode, search_term):
    options = sorted(ROOT.glob(f'{search_term}/{mode}/summary.csv'))
    if len(options) != 1:
        raise FileNotFoundError(f'Expected one summary.csv under pls_mappo/{search_term}/{mode}/')
    frame = pd.read_csv(options[0])
    frame = frame[(frame['status'] == 'ok') & (frame['mode'] == mode)].copy()
    frame = frame.drop_duplicates(subset=['case_id', 'policy'], keep='last')
    if frame.empty:
        raise ValueError('No successful evaluations')
    return frame


def plot_network_or_traffic(which):
    if which == 'uavs':
        df = find_summary('scalability', 'research_gaps_fixed_iot12')
        assert (df['num_iot'] == 12).all(), 'IoT count is not fixed!'
        key = 'num_uavs'; xlabel = 'Number of UAVs (12 IoT devices fixed)'
        prefix = '23_fixed_iot_uavs'
    else:
        df = find_summary('traffic', 'research_gaps_tx_demand')
        assert (df['num_uavs'] == 3).all() and (df['num_iot'] == 12).all()
        key = 'requested_tx_w'; xlabel = 'Requested IoT transmit power (W) — demand proxy'
        prefix = '25_traffic_demand'
    for metric, ylabel, suffix in (
        ('secrecy_mbps', 'Mean Secrecy Rate (Mbps)', 'secrecy'),
        ('secrecy_mbits_per_kj', 'Secure Energy Efficiency (Mbits/kJ)', 'efficiency'),
        ('final_depleted_iot_fraction', 'Final Depleted IoT Fraction', 'depletion'),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for policy in ('mappo', 'random', 'stationary'):
            subset = df[df['policy'] == policy].sort_values(key)
            if subset.empty:
                continue
            group = subset.groupby(key, as_index=False).agg(
                mean=(metric + '_mean', 'mean'))
            ax.plot(group[key], group['mean'], marker='o', label=policy.upper())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ' vs ' + xlabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        destination = OUT / f'{prefix}_{suffix}.png'
        fig.savefig(destination, dpi=300)
        plt.close(fig)
        print('Saved:', destination)
    print('NOTE: 10-update/one-seed comparisons are preliminary only.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--what', choices=['lr', 'uavs', 'traffic'], required=True)
    args = p.parse_args()
    plot_lr() if args.what == 'lr' else plot_network_or_traffic(args.what)
