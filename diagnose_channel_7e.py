"""Stage 7E: calibrated channel diagnostics without modifying the project.
Run from project root: python diagnose_channel_stage7e.py
"""
import json
from pathlib import Path
import numpy as np
from pls_mappo.channel_model import RISChannelModel
from pls_mappo.config import SimulationConfig
from pls_mappo.energy_model import EnergyParameters, harvested_energy_j

cfg = SimulationConfig()
e = EnergyParameters()
ris = np.asarray(cfg.ris_position, dtype=float)
source = np.array([100., 100., cfg.uav_altitude_m])
rows = []
for horizontal_distance in [0, 20, 50, 100]:
    dest = np.array([100. + horizontal_distance, 100., 0.])
    model = RISChannelModel(cfg, seed=2026 + horizontal_distance)
    stats = {'direct_only': [], 'ris_zero': [], 'ris_ideal': []}
    for trial in range(1000):
        # Use identical sampled paths to compare RIS conditions fairly.
        direct = model._link_vector(source, dest)[0]
        s2r = model._link_vector(source, ris, cfg.num_ris_elements)
        r2d = model._link_vector(ris, dest, cfg.num_ris_elements)
        cascaded = s2r * r2d
        zero = direct + cascaded.sum()
        ideal = abs(direct) + np.abs(cascaded).sum()
        stats['direct_only'].append(abs(direct)**2)
        stats['ris_zero'].append(abs(zero)**2)
        stats['ris_ideal'].append(ideal**2)
    means = {k: float(np.mean(v)) for k,v in stats.items()}
    d = float(np.linalg.norm(source-dest))
    harvest = harvested_energy_j(means['ris_zero'], 0.5, cfg.slot_duration_s, e)
    demanded = (0.05 + e.iot_circuit_power_w) * 0.5 * cfg.slot_duration_s
    rows.append({'horizontal_distance_m':horizontal_distance,
                 'slant_distance_m':d,
                 'mean_gain_direct':means['direct_only'],
                 'mean_gain_ris_zero_phase':means['ris_zero'],
                 'mean_gain_ris_ideal_phase_upper_bound':means['ris_ideal'],
                 'mean_harvest_uJ_at_half_slot':float(harvest*1e6),
                 'consumption_uJ_at_half_slot':float(demanded*1e6),
                 'harvest_to_consumption_ratio':float(harvest/demanded)})
print('Distance | Direct gain | RIS zero gain | Ideal-phase upper gain | Harvest uJ/slot | Needed uJ/slot | Harvest/demand')
for r in rows:
    print(f"{r['horizontal_distance_m']:>7}m | {r['mean_gain_direct']:.3e} | {r['mean_gain_ris_zero_phase']:.3e} | {r['mean_gain_ris_ideal_phase_upper_bound']:.3e} | {r['mean_harvest_uJ_at_half_slot']:.4f} | {r['consumption_uJ_at_half_slot']:.0f} | {r['harvest_to_consumption_ratio']:.3e}")
out=Path('pls_mappo/results_stage7e/channel_diagnostics.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
print('Saved:',out)
