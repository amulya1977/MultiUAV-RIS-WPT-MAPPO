"""Stage 7G: finite-horizon battery feasibility diagnostics (NO training).
Run from repository root: python diagnose_battery_stage7g.py
Tests stationary UAVs with multiple WPT fractions and IoT power requests.
Does not change any simulation parameters or reward settings.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pls_mappo.environment import PLSMultiUAVEnv


def run_case(seed, steps, fraction, tx_power):
    env = PLSMultiUAVEnv(episode_steps=steps, requested_iot_tx_w=tx_power)
    env.reset(seed=seed)
    initial = float(np.sum(env.batteries))
    total_harvest = total_consumption = total_battery_change = total_secrecy = 0.0
    total_active = total_requested_tx_energy = total_uav_energy = 0.0
    conservation_error = 0.0
    first_depletion_step = None
    never_depleted = True
    for t in range(steps):
        actions = np.zeros((env.n_agents, env.action_dim), dtype=float)
        actions[:, 3] = 2 * fraction - 1
        # Hold UAV position, 50% AN, fixed RIS phases at pi.
        old = env.batteries.copy()
        _, _, _, done, info = env.step(actions)
        new = env.batteries.copy()
        harvested = float(info['harvested_energy_j'])
        used = float(info['iot_transmission_energy_j'])
        changed = float(info['net_iot_battery_change_j'])
        # Finite battery capacity can discard harvested RF energy. Infer discarded
        # energy from the conservation equation instead of assuming zero overflow.
        overflow = harvested - used - changed
        assert overflow >= -1e-8, f'Energy creation beyond numerical tolerance: {overflow}'
        assert np.all((new >= -1e-10) & (new <= env.energy.battery_capacity_j + 1e-10))
        assert abs(changed - float(np.sum(new-old))) < 1e-8
        conservation_error = max(conservation_error, abs(harvested-used-changed-max(overflow,0)))
        total_harvest += harvested
        total_consumption += used
        total_battery_change += changed
        total_active += info['active_iot_count']
        total_secrecy += float(info['secrecy_rate_bps'])
        total_uav_energy += float(info['uav_energy_j'])
        # Upper bound on demand for every active device at requested TX power.
        total_requested_tx_energy += info['active_iot_count'] * (tx_power + env.energy.iot_circuit_power_w) * (1-fraction) * env.cfg.slot_duration_s
        depleted = bool(np.any(new <= 1e-6))
        if depleted and first_depletion_step is None:
            first_depletion_step = t + 1
        never_depleted &= not depleted
        if done:
            break
    final = float(np.sum(env.batteries))
    assert abs((final-initial) - total_battery_change) < 1e-7
    ratio = total_harvest / total_consumption if total_consumption > 0 else None
    return {
        'seed': seed, 'steps': steps, 'harvesting_fraction': fraction,
        'requested_tx_power_w': tx_power, 'initial_total_battery_j': initial,
        'final_total_battery_j': final, 'mean_final_battery_j': final/env.cfg.num_iot,
        'total_harvested_j': total_harvest, 'total_consumed_j': total_consumption,
        'total_net_battery_change_j': total_battery_change,
        'harvest_to_consumption_ratio': ratio,
        'energy_neutral_over_horizon': bool(total_harvest >= total_consumption and total_battery_change >= -1e-8),
        'never_depleted_any_device': never_depleted,
        'first_depleted_step': first_depletion_step,
        'harvest_j_per_active_device_slot': total_harvest/max(total_active,1),
        'consumed_j_per_active_device_slot': total_consumption/max(total_active,1),
        'requested_energy_served_fraction': min(1.,total_consumption/max(total_requested_tx_energy,1e-15)),
        'mean_sum_secrecy_mbps': total_secrecy/steps/1e6,
        'mean_uav_energy_j_per_step': total_uav_energy/steps,
        'max_energy_conservation_residual_j': conservation_error,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    p.add_argument('--fractions',type=float,nargs='+',default=[0.25,0.5,0.95])
    p.add_argument('--powers',type=float,nargs='+',default=[0.05,0.005,0.0005])
    p.add_argument('--output-dir',default='pls_mappo/results_stage7g')
    args=p.parse_args()
    if args.steps<1 or not args.seeds or not args.fractions or not args.powers:
        p.error('positive steps and nonempty groups required')
    if any(not 0<=f<=1 for f in args.fractions) or any(not 0<=x<=0.1 for x in args.powers):
        p.error('fractions must be in [0,1], tx powers in [0,0.1]')
    rows=[run_case(seed,args.steps,f,power) for power in args.powers for f in args.fractions for seed in args.seeds]
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'battery_feasibility.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (out/'battery_feasibility.csv').open('w',encoding='utf-8',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Stage 7G battery feasibility (each row is an independent simulation)')
    print('seed  WPT   TX(W)   harvest/use  final battery/device (J)  first depleted step  energy-neutral')
    for r in rows:
        ratio=r['harvest_to_consumption_ratio']
        print(f"{r['seed']:4} {r['harvesting_fraction']:4.2f} {r['requested_tx_power_w']:7.4f} "
              f"{ratio if ratio is not None else float('nan'):11.3e} {r['mean_final_battery_j']:23.6f} "
              f"{str(r['first_depleted_step']):>19} {str(r['energy_neutral_over_horizon']):>14}")
    print(f'Validated maximum energy conservation residual: {max(r["max_energy_conservation_residual_j"] for r in rows):.3e} J')
    print(f'Results saved to {out}')

if __name__=='__main__':
    main()
