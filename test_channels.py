import numpy as np
from pls_mappo.config import SimulationConfig
from pls_mappo.channel_model import RISChannelModel

cfg = SimulationConfig()
model = RISChannelModel(cfg)
uav = [30.0, 30.0, cfg.uav_altitude_m]
device = [80.0, 60.0, 0.0]
phases = np.zeros(cfg.num_ris_elements)
for step in range(5):
    gain = model.channel_gain(uav, device, phases)
    assert np.isfinite(gain) and gain >= 0
    print(f'Step {step + 1}: RIS-assisted channel gain = {gain:.8e}')
print('Channel model test PASSED')
