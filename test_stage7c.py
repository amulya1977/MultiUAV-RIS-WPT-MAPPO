import numpy as np
from pls_mappo.environment import PLSMultiUAVEnv


def main():
    env = PLSMultiUAVEnv(episode_steps=5, collision_penalty=0.2,
                         charging_deficit_penalty=0.1,
                         target_harvest_j_per_active_iot=1e-5)
    obs, state = env.reset(seed=42)
    assert obs.shape == (env.n_agents, env.obs_dim)
    assert env.obs_dim == 54, env.obs_dim
    assert np.isfinite(obs).all()
    a = np.zeros((env.n_agents, env.action_dim))
    for _ in range(5):
        obs, state, reward, done, info = env.step(a)
        assert np.isfinite(reward).all()
        assert np.isfinite(obs).all()
        assert 0 <= info['collision_reverted'] <= env.n_agents
        assert 0 <= info['collision_reverted_fraction'] <= 1
        assert np.min(np.linalg.norm(env.positions[:, None, :] - env.positions[None, :, :] +
                                     np.eye(env.n_agents)[:, :, None] * 1000, axis=2)) >= env.cfg.min_uav_separation_m - 1e-8
    assert done
    # Deliberately request overlapping trajectories. The safety filter must reject moves.
    env.reset(seed=42)
    env.positions[0, :2] = (50, 50)
    env.positions[1, :2] = (64, 50)
    env.positions[2, :2] = (150, 150)
    a[:] = 0
    a[0, 0] = 1
    a[1, 0] = -1
    _, _, _, _, info = env.step(a)
    assert info['collision_reverted'] >= 1, info
    assert np.linalg.norm(env.positions[0] - env.positions[1]) >= env.cfg.min_uav_separation_m
    print('Stage 7C diagnostics and swept-separation tests PASSED')

if __name__ == '__main__':
    main()
