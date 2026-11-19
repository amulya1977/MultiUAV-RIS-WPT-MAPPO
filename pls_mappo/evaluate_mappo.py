"""Stage 7I: matched-seed checkpoint and baseline evaluation (no training).
Run from repository root: python -m pls_mappo.evaluate_mappo --checkpoints ...
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from pls_mappo.environment import PLSMultiUAVEnv
from pls_mappo.config import SimulationConfig
from pls_mappo.train_mappo import SharedActor

METRICS = (
    'reward_per_step', 'secrecy_mbps', 'uav_energy_j_per_step',
    'secrecy_mbits_per_kj', 'collision_reversions_per_step',
    'actual_flight_distance_m_per_step', 'attempted_flight_distance_m_per_step',
    'mean_iot_battery_j', 'final_mean_iot_battery_j',
    'final_min_iot_battery_j', 'depleted_iot_fraction',
    'final_depleted_iot_fraction', 'harvested_j_per_step',
    'iot_energy_consumed_j_per_step', 'harvest_to_consumption_ratio',
    'net_iot_battery_change_j_per_step', 'active_iot_per_step',
)


def make_env(args):
    return PLSMultiUAVEnv(
        config=SimulationConfig(num_uavs=args.num_uavs, num_iot=args.num_iot),
        episode_steps=args.episode_steps,
        requested_iot_tx_w=args.iot_tx_power_w,
        collision_penalty=args.collision_penalty,
        charging_deficit_penalty=args.charging_deficit_penalty,
        target_harvest_j_per_active_iot=args.target_harvest_j,
    )


def load_actor(path, env, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError('Requires a PyTorch version supporting weights_only=True') from exc
    for key, expected in [('obs_dim', env.obs_dim), ('action_dim', env.action_dim),
                          ('n_agents', env.n_agents), ('state_dim', len(env._state()))]:
        if int(checkpoint[key]) != expected:
            raise ValueError(f'{path}: {key} is {checkpoint[key]}, expected {expected}. '
                             'Use a matching environment and checkpoint version.')
    hidden = checkpoint['actor']['body.0.weight'].shape[0]
    actor = SharedActor(env.obs_dim, env.action_dim, hidden=hidden).to(device)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    return actor


def choose_actions(label, env, obs, actor, rng, device):
    if actor is not None:
        with torch.no_grad():
            dist = actor.distribution(torch.as_tensor(obs, dtype=torch.float32, device=device))
            return torch.tanh(dist.loc).cpu().numpy()
    if label == 'random':
        return rng.uniform(-1.0, 1.0, size=(env.n_agents, env.action_dim))
    if label == 'stationary':
        actions = np.zeros((env.n_agents, env.action_dim), dtype=float)
        actions[:, 3] = 0.0  # 50% WPT harvesting
        actions[:, 4] = -1.0  # no artificial noise
        actions[:, 5:] = -1.0  # zero RIS phase, following environment action mapping
        return actions
    raise ValueError(label)


def evaluate(label, actor, seed, args, device):
    env = make_env(args)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 100000)
    sums = {key: 0.0 for key in (
        'reward', 'secrecy_mbits', 'uav_energy_j', 'collision_reversions',
        'actual_distance_m', 'attempted_distance_m', 'mean_iot_battery_j',
        'depleted_iot_fraction', 'harvested_j', 'iot_consumed_j',
        'battery_change_j', 'active_iot',
    )}
    info = None
    for step in range(args.episode_steps):
        actions = choose_actions(label, env, obs, actor, rng, device)
        obs, _, rewards, done, info = env.step(actions)
        dt = env.cfg.slot_duration_s
        sums['reward'] += float(np.mean(rewards))
        sums['secrecy_mbits'] += float(info['secrecy_rate_bps']) * dt / 1e6
        sums['uav_energy_j'] += float(info['uav_energy_j'])
        sums['collision_reversions'] += float(info['collision_reverted'])
        sums['actual_distance_m'] += float(info['actual_distance_m'])
        sums['attempted_distance_m'] += float(info['attempted_distance_m'])
        sums['mean_iot_battery_j'] += float(info['mean_iot_battery_j'])
        sums['depleted_iot_fraction'] += float(info['depleted_iot_fraction'])
        sums['harvested_j'] += float(info['harvested_energy_j'])
        sums['iot_consumed_j'] += float(info['iot_transmission_energy_j'])
        sums['battery_change_j'] += float(info['net_iot_battery_change_j'])
        sums['active_iot'] += float(info['active_iot_count'])
        if done:
            break
    steps = env.t
    return dict(
        policy=label, seed=seed, steps=steps,
        reward_per_step=sums['reward']/steps,
        secrecy_mbps=sums['secrecy_mbits']/(steps * env.cfg.slot_duration_s),
        uav_energy_j_per_step=sums['uav_energy_j']/steps,
        secrecy_mbits_per_kj=(1000*sums['secrecy_mbits']/sums['uav_energy_j']) if sums['uav_energy_j'] > 0 else 0.,
        collision_reversions_per_step=sums['collision_reversions']/steps,
        actual_flight_distance_m_per_step=sums['actual_distance_m']/steps,
        attempted_flight_distance_m_per_step=sums['attempted_distance_m']/steps,
        mean_iot_battery_j=sums['mean_iot_battery_j']/steps,
        final_mean_iot_battery_j=float(info['mean_iot_battery_j']),
        final_min_iot_battery_j=float(info['min_iot_battery_j']),
        depleted_iot_fraction=sums['depleted_iot_fraction']/steps,
        final_depleted_iot_fraction=float(info['depleted_iot_fraction']),
        harvested_j_per_step=sums['harvested_j']/steps,
        iot_energy_consumed_j_per_step=sums['iot_consumed_j']/steps,
        harvest_to_consumption_ratio=sums['harvested_j']/max(sums['iot_consumed_j'],1e-15),
        net_iot_battery_change_j_per_step=sums['battery_change_j']/steps,
        active_iot_per_step=sums['active_iot']/steps,
    )


def summarize(rows, labels):
    result = {}
    for label in labels:
        subset = [r for r in rows if r['policy'] == label]
        result[label] = {}
        for key in METRICS:
            vals = np.array([r[key] for r in subset], dtype=float)
            result[label][key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
    return result


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable: skipping plots')
        return
    plots = [('secrecy_mbps', 'Aggregate secrecy (Mbps)', 'secrecy_comparison.png'),
             ('uav_energy_j_per_step', 'UAV energy (J/step)', 'uav_energy_comparison.png'),
             ('secrecy_mbits_per_kj', 'Secrecy efficiency (Mbits/kJ)', 'efficiency_comparison.png'),
             ('final_mean_iot_battery_j', 'Final mean IoT battery (J)', 'battery_comparison.png'),
             ('collision_reversions_per_step', 'Rejected UAV moves per step', 'collision_comparison.png')]
    labels = list(summary)
    for key, ylabel, filename in plots:
        means = [summary[label][key]['mean'] for label in labels]
        stds = [summary[label][key]['std'] for label in labels]
        fig, ax = plt.subplots(figsize=(max(7, len(labels)*1.3), 4))
        ax.bar(range(len(labels)), means, yerr=stds, capsize=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title('Matched held-out seeds: mean ± sample SD')
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Stage 7I matched-seed checkpoint evaluation')
    p.add_argument('--checkpoints', nargs='+', required=True, help='Checkpoint .pt files to compare')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--episode-steps', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=2000)
    p.add_argument('--iot-tx-power-w', type=float, default=0.005)
    p.add_argument('--num-uavs', type=int, default=3)
    p.add_argument('--num-iot', type=int, default=12)
    p.add_argument('--collision-penalty', type=float, default=0.2)
    p.add_argument('--charging-deficit-penalty', type=float, default=0.0)
    p.add_argument('--target-harvest-j', type=float, default=0.0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-dir', default='pls_mappo/results_stage7i_eval')
    args = p.parse_args()
    if args.episodes < 1 or args.episode_steps < 1:
        p.error('episodes and episode-steps must be positive')
    if not 0 <= args.iot_tx_power_w <= 0.1:
        p.error('iot-tx-power-w must be between 0 and 0.1 W')
    for path in args.checkpoints:
        if not Path(path).is_file():
            p.error(f'Checkpoint missing: {path}')
    if len(set(args.checkpoints)) != len(args.checkpoints):
        p.error('Do not repeat checkpoint paths')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prototype = make_env(args)
    prototype.reset(seed=args.seed_start)
    device = torch.device(args.device)
    policies = []
    for path in args.checkpoints:
        checkpoint = Path(path)
        label = f'{checkpoint.parent.name}_{checkpoint.stem}'
        policies.append((label, load_actor(checkpoint, prototype, device)))
    policies += [('random', None), ('stationary', None)]
    rows = []
    for label, actor in policies:
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            rows.append(evaluate(label, actor, seed, args, device))
        sample = [r for r in rows if r['policy'] == label]
        print(f'{label}: secrecy={np.mean([r["secrecy_mbps"] for r in sample]):.3f} Mbps | '
              f'energy={np.mean([r["uav_energy_j_per_step"] for r in sample]):.1f} J/step | '
              f'final battery={np.mean([r["final_mean_iot_battery_j"] for r in sample]):.4f} J', flush=True)
    summary = summarize(rows, [label for label, _ in policies])
    save_csv(output / 'episodes.csv', rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'evaluation_config.json').write_text(json.dumps({
        **vars(args), 'evaluation_action_mode': 'deterministic_actor_mean',
        'comparison': 'matched seed indexes; stochastic trajectories can diverge by policy',
        'battery_sustainability': 'not guaranteed; 100-slot finite-horizon evaluation',
        'demand_served': 'not measured by current environment info dictionary',
    }, indent=2), encoding='utf-8')
    plot_summary(summary, output)
    print(f'Finished: {output} (episodes.csv, summary.json, plots)')


if __name__ == '__main__':
    main()
