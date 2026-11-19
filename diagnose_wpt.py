"""Stage 7D: standalone, non-training WPT/battery feasibility diagnostic.
Run from the project root: python diagnose_wpt.py
Writes results_stage7d/wpt_feasibility.csv and .json.
"""
import csv
import json
from pathlib import Path
import numpy as np
from pls_mappo.environment import PLSMultiUAVEnv

SEED = 42
EPISODE_STEPS = 100
HARVEST_FRACTIONS = (0.25, 0.50, 0.75, 0.95)
REQUESTED_TX_W = (0.05, 0.005, 0.0005)


def run_case(fraction, tx_w):
    env = PLSMultiUAVEnv(episode_steps=EPISODE_STEPS, requested_iot_tx_w=tx_w)
    env.reset(seed=SEED)
    initial_batteries = env.batteries.copy()
    harvested_total = 0.0
    transmitted_total = 0.0
    active_total = 0
    battery_sum = 0.0
    depleted_active = 0
    uav_energy_total = 0.0
    secrecy_total = 0.0
    steps = 0
    for _ in range(EPISODE_STEPS):
        actions = np.zeros((env.n_agents, env.action_dim), dtype=float)
        actions[:, :2] = 0.0  # Stationary UAVs, for controlled feasibility test
        actions[:, 3] = 2 * fraction - 1
        actions[:, 4] = 0.0  # AN ratio 0.5 by default
        actions[:, 5:] = 0.0  # RIS phase pi
        _, _, _, done, info = env.step(actions)
        steps += 1
        harvested_total += info['harvested_energy_j']
        transmitted_total += info['iot_transmission_energy_j']
        active_total += info['active_iot_count']
        battery_sum += info['mean_iot_battery_j']
        uav_energy_total += info['uav_energy_j']
        secrecy_total += info['secrecy_rate_bps']
        depleted_active += int(np.sum(env.iot_active & (env.batteries < 1e-5)))
        if done:
            break
    return dict(
        harvesting_fraction=fraction,
        requested_tx_power_w=tx_w,
        steps=steps,
        initial_battery_j=float(np.mean(initial_batteries)),
        final_mean_battery_j=float(np.mean(env.batteries)),
        mean_battery_j=battery_sum / steps,
        harvested_j_per_active_iot_slot=harvested_total / max(active_total, 1),
        consumed_j_per_active_iot_slot=transmitted_total / max(active_total, 1),
        harvested_to_consumed_ratio=harvested_total / transmitted_total if transmitted_total > 0 else None,
        depleted_active_device_fraction=depleted_active / max(active_total, 1),
        mean_uav_energy_j_per_step=uav_energy_total / steps,
        mean_sum_secrecy_mbps=secrecy_total / steps / 1e6,
    )


def main():
    results = [run_case(f, p) for p in REQUESTED_TX_W for f in HARVEST_FRACTIONS]
    out = Path('pls_mappo/results_stage7d')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'wpt_feasibility.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (out / 'wpt_feasibility.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('frac  tx(W)   harvested/active(J)  consumed/active(J)  final battery(J)  depletion fraction')
    for r in results:
        print(f"{r['harvesting_fraction']:4.2f}  {r['requested_tx_power_w']:6.4f}  "
              f"{r['harvested_j_per_active_iot_slot']:19.3e}  "
              f"{r['consumed_j_per_active_iot_slot']:18.3e}  "
              f"{r['final_mean_battery_j']:16.5f}  "
              f"{r['depleted_active_device_fraction']:17.3f}")
    print(f'\nResults saved in {out}')


if __name__ == '__main__':
    main()
