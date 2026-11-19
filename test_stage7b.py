"""Stage 7B regression and reward-penalty checks."""
import numpy as np
from pls_mappo.environment import PLSMultiUAVEnv


def rollout(env, seed, actions):
    env.reset(seed=seed)
    outputs = []
    for action in actions:
        _, _, r, _, info = env.step(action)
        outputs.append((r.copy(), info))
    return outputs

base = PLSMultiUAVEnv()
K, D = base.n_agents, base.action_dim
rng = np.random.default_rng(17)
actions = [rng.uniform(-1, 1, (K, D)) for _ in range(8)]
legacy = rollout(PLSMultiUAVEnv(), 101, actions)
zero = rollout(PLSMultiUAVEnv(collision_penalty=0., charging_deficit_penalty=0.,
                             target_harvest_j_per_active_iot=0.), 101, actions)
for (r0, i0), (r1, i1) in zip(legacy, zero):
    np.testing.assert_allclose(r0, r1, rtol=0, atol=0)
    assert i0['collision_reverted'] == i1['collision_reverted']
    assert np.isfinite(i0['attempted_distance_m'])
    assert i0['actual_distance_m'] >= 0
    assert 0 <= i0['min_iot_battery_j'] <= i0['mean_iot_battery_j'] <= 2.0

penalized = rollout(PLSMultiUAVEnv(collision_penalty=0.2,
                                   charging_deficit_penalty=0.1,
                                   target_harvest_j_per_active_iot=1e-5), 101, actions)
for (old_r, old_i), (new_r, new_i) in zip(legacy, penalized):
    assert np.all(new_r <= old_r + 1e-6)
    assert np.all(np.isfinite(new_r))
    assert new_i['charging_deficit_j'] >= 0
    assert new_i['collision_reverted'] == old_i['collision_reverted']
    assert np.allclose(new_i['uav_positions_m'], old_i['uav_positions_m'])
assert any(np.any(x[0] < y[0] - 1e-8) for x,y in zip(penalized,legacy))
print('Stage 7B reward and diagnostics tests PASSED')
