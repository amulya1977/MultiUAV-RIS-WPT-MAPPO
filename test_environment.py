import numpy as np
from pls_mappo.environment import PLSMultiUAVEnv


def rollout(seed):
    env = PLSMultiUAVEnv(episode_steps=10)
    obs, state = env.reset(seed=seed)
    assert obs.shape == (env.n_agents, env.obs_dim), obs.shape
    assert state.ndim == 1 and np.isfinite(state).all()
    assert env.action_dim == 9, env.action_dim
    rng = np.random.default_rng(seed + 10)
    logs = []
    for step in range(10):
        actions = rng.uniform(-1, 1, size=(env.n_agents, env.action_dim))
        obs, new_state, rewards, done, info = env.step(actions)
        assert obs.shape == (env.n_agents, env.obs_dim)
        assert state.shape == new_state.shape
        assert rewards.shape == (env.n_agents,)
        assert np.isfinite(obs).all() and np.isfinite(rewards).all()
        assert info['secrecy_rate_bps'] >= 0 and info['harvested_energy_j'] >= 0
        assert info['uav_energy_j'] >= 0
        assert np.all((env.batteries >= 0) & (env.batteries <= env.energy.battery_capacity_j))
        assert info['max_uav_speed_mps'] <= env.cfg.max_uav_speed_mps + 1e-8
        for k in range(env.n_agents):
            for m in range(k+1, env.n_agents):
                assert np.linalg.norm(env.positions[k] - env.positions[m]) >= env.cfg.min_uav_separation_m
        assert done == (step == 9)
        logs.append((info['secrecy_rate_bps'], info['uav_energy_j']))
        if step < 3:
            print(f"Step {step+1}: secrecy={info['secrecy_rate_bps']:.2f} bps; "
                  f"UAV energy={info['uav_energy_j']:.2f} J; active IoT={info['active_iot_count']}")
        state = new_state
    return np.array(logs)


if __name__ == '__main__':
    first, second = rollout(42), rollout(42)
    assert np.array_equal(first, second), 'Rollout must be reproducible with same seed'
    print('Stage 5 multi-UAV environment test PASSED')
