import numpy as np
from pls_mappo.config import SimulationConfig
from pls_mappo.channel_model import RISChannelModel
from pls_mappo.energy_model import (EnergyParameters, harvested_energy_j,
    iot_battery_step, uav_propulsion_power_w, uav_slot_energy_j)

cfg = SimulationConfig()
model = RISChannelModel(cfg)
p = EnergyParameters()
uav = [30.0, 30.0, cfg.uav_altitude_m]
device = [80.0, 60.0, 0.0]
gain = model.channel_gain(uav, device, np.zeros(cfg.num_ris_elements))
harvested = harvested_energy_j(gain, 0.4, cfg.slot_duration_s, p)
state = iot_battery_step(p.initial_battery_j, harvested, 0.4,
                         cfg.slot_duration_s, 0.05, p)
uav_energy = uav_slot_energy_j([3.0, 4.0], 0.4, 0.2,
                                   cfg.slot_duration_s, [0.0, 0.0], p)
assert harvested >= 0 and np.isfinite(harvested)
assert 0 <= state['battery_j'] <= p.battery_capacity_j
assert state['tx_energy_j'] + state['circuit_energy_j'] <= min(p.battery_capacity_j, p.initial_battery_j + harvested) + 1e-12
assert abs(uav_energy['total_j'] - sum(uav_energy[k] for k in ('propulsion_j','wpt_j','artificial_noise_j'))) < 1e-9
assert uav_propulsion_power_w([0, 0], params=p) > 0
# No energy to transmit if battery is empty and no WPT received.
empty = iot_battery_step(0, 0, 0.4, 1, 0.05, p)
assert empty['tx_power_w'] == 0 and empty['battery_j'] == 0
print(f'Channel power gain: {gain:.8e}')
print(f'Harvested energy: {harvested:.8e} J')
print(f'IoT transmit power: {state["tx_power_w"]:.6f} W')
print(f'IoT battery after slot: {state["battery_j"]:.6f} J')
print(f'UAV propulsion energy: {uav_energy["propulsion_j"]:.3f} J')
print(f'UAV WPT energy: {uav_energy["wpt_j"]:.3f} J')
print(f'UAV artificial-noise energy: {uav_energy["artificial_noise_j"]:.3f} J')
print(f'UAV total energy: {uav_energy["total_j"]:.3f} J')
print('Energy model test PASSED')
