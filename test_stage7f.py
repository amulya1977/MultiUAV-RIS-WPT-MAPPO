import numpy as np
from pls_mappo.channel_model import RISChannelModel
from pls_mappo.config import SimulationConfig
from pls_mappo.environment import PLSMultiUAVEnv


def run():
    cfg = SimulationConfig()
    m = RISChannelModel(cfg, seed=123)
    x, y = (20., 20., 40.), (80., 70., 0.)
    p = np.zeros(cfg.num_ris_elements)
    g1 = m.channel_gain(x, y, p)
    assert g1 == m.channel_gain(x, y, p), 'channel varies inside one slot'
    assert np.isclose(g1, m.channel_gain(y, x, p)), 'reciprocity failed'
    direct, elements = m.channel_components(x, y)
    assert np.isclose(g1, abs(direct + np.sum(elements)) ** 2)
    phase_aligned = (np.angle(direct) - np.angle(elements)) % (2*np.pi)
    optimized = m.channel_gain(x, y, phase_aligned)
    assert optimized >= g1 - 1e-15, 'ideal phase alignment should not decrease gain'
    m.begin_slot()
    assert m.channel_gain(x, y, p) == m.channel_gain(x, y, p)
    a = PLSMultiUAVEnv(episode_steps=4)
    a.reset(seed=42)
    actions = np.zeros((a.n_agents, a.action_dim), dtype=float)
    for _ in range(4):
        _, _, reward, _, info = a.step(actions)
        assert np.isfinite(reward).all()
        assert info['active_iot_harvest_to_consumption_ratio'] >= 0
        assert 0 <= info['depleted_iot_fraction'] <= 1
        assert np.isclose(info['net_iot_battery_change_j'],
                          info['harvested_energy_j'] - info['iot_transmission_energy_j'],
                          atol=2e-6), 'battery energy conservation mismatch'
    print('Stage 7F: channel reuse, reciprocity and energy diagnostics PASSED')

if __name__ == '__main__':
    run()
