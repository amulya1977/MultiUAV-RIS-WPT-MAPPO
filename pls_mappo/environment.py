"""Stage 5: integrated, NumPy-only multi-UAV RIS/WPT/PLS environment.

API:
  obs, state = env.reset(seed=...)
  next_obs, next_state, rewards, done, info = env.step(actions)

Prototype choices (not paper equivalence): fixed UAV altitude; a reserved altitude
control is ignored; per-slot fading is resampled; RIS elements are partitioned
by Kmax, with unused elements held at zero; QoS and depot-return are diagnostic
only. Reward units are normalized Mbps and ~150 J, not raw SI units.
"""
import numpy as np
from .config import SimulationConfig
from .channel_model import RISChannelModel
from .energy_model import EnergyParameters, harvested_energy_j, iot_battery_step, uav_slot_energy_j
from .security_model import secrecy_rate_bps


class PLSMultiUAVEnv:
    def __init__(self, config=None, energy_params=None, episode_steps=100,
                 bandwidth_hz=1_000_000., noise_power_w=1e-10,
                 requested_iot_tx_w=0.05, max_an_ratio=1.0,
                 collision_penalty=0.0, charging_deficit_penalty=0.0,
                 target_harvest_j_per_active_iot=0.0):
        self.cfg = config or SimulationConfig()
        self.energy = energy_params or EnergyParameters()
        self.episode_steps = int(episode_steps)
        if self.episode_steps < 1 or bandwidth_hz <= 0 or noise_power_w <= 0:
            raise ValueError('Invalid episode/radio parameters')
        if not 0 <= requested_iot_tx_w <= self.energy.iot_max_tx_power_w:
            raise ValueError('Requested transmit power exceeds IoT maximum')
        if not 0 <= max_an_ratio <= 1:
            raise ValueError('max_an_ratio must be in [0,1]')
        for name, val in [('collision_penalty', collision_penalty),
                          ('charging_deficit_penalty', charging_deficit_penalty),
                          ('target_harvest_j_per_active_iot', target_harvest_j_per_active_iot)]:
            if not np.isfinite(val) or val < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        self.collision_penalty = float(collision_penalty)
        self.charging_deficit_penalty = float(charging_deficit_penalty)
        self.target_harvest_j_per_active_iot = float(target_harvest_j_per_active_iot)
        self.bandwidth_hz = bandwidth_hz
        self.noise_power_w = noise_power_w
        self.requested_iot_tx_w = requested_iot_tx_w
        self.max_an_ratio = max_an_ratio
        self.n_agents = self.cfg.num_uavs
        self.ris_elements_per_agent = self.cfg.num_ris_elements // self.cfg.max_uavs
        self.action_dim = 5 + self.ris_elements_per_agent
        # 3 pos + 3 velocity + 4 capability + (Kmax-1)*4 other UAV values
        # + Lk previous phases + 3 last feedback + 1 time + 8 IoT/safety features
        self.obs_dim = 6 + 4 + 4 * (self.cfg.max_uavs - 1) + self.ris_elements_per_agent + 3 + 1 + 8
        self.rng = None
        self.channel_model = None
        self.reset()

    def reset(self, seed=None):
        rng_seed = self.cfg.seed if seed is None else seed
        self.rng = np.random.default_rng(rng_seed)
        self.channel_model = RISChannelModel(self.cfg, seed=int(self.rng.integers(0, 2**31-1)))
        A, K, N, J = self.cfg.area_size_m, self.n_agents, self.cfg.num_iot, self.cfg.num_eavesdroppers
        # Deliberately spaced depot positions; deterministic across resets.
        self.depots = np.stack([np.array([A * (k+1)/(K+1), A * 0.25]) for k in range(K)])
        self.positions = np.column_stack((self.depots.copy(), np.full(K, self.cfg.uav_altitude_m)))
        self.velocities = np.zeros((K, 2), dtype=float)
        self.iot_positions = np.column_stack((self.rng.uniform(0, A, (N, 2)), np.zeros(N)))
        self.eaves_positions = np.column_stack((self.rng.uniform(0, A, (J, 2)), np.zeros(J)))
        self.batteries = np.full(N, self.energy.initial_battery_j, dtype=float)
        self.iot_active = np.ones(N, dtype=bool)
        self.ris_phases = np.zeros(self.cfg.num_ris_elements, dtype=float)
        self.previous_feedback = np.zeros((K, 3), dtype=float)
        self.previous_collision = np.zeros(K, dtype=float)
        self.t = 0
        return self._observations(), self._state()

    def _observations(self):
        K, Km, A = self.n_agents, self.cfg.max_uavs, self.cfg.area_size_m
        out = []
        for k in range(K):
            own = np.r_[self.positions[k, :2] / A,
                        self.positions[k, 2] / max(self.cfg.uav_altitude_m, 1),
                        self.velocities[k] / self.cfg.max_uav_speed_mps, 0.0]
            capabilities = np.array([self.cfg.max_uav_speed_mps / 20,
                                     self.energy.wpt_power_w / 20,
                                     self.energy.uav_max_an_power_w / 2,
                                     self.energy.battery_capacity_j / 5])
            others = []
            for m in range(Km):
                if m < K and m != k:
                    others.extend([self.positions[m, 0] / A, self.positions[m, 1] / A,
                                   self.positions[m, 2] / max(self.cfg.uav_altitude_m, 1), 1.0])
                elif m != k:
                    others.extend([0., 0., 0., 0.])
            sl = slice(k * self.ris_elements_per_agent, (k+1) * self.ris_elements_per_agent)
            distances = np.linalg.norm(self.iot_positions[:, None, :2] -
                                       self.positions[None, :, :2], axis=2)
            associated = np.argmin(distances, axis=1)
            own_ids = np.flatnonzero((associated == k) & self.iot_active)
            if len(own_ids):
                nearest = own_ids[np.argmin(distances[own_ids, k])]
                nearest_delta = (self.iot_positions[nearest, :2] - self.positions[k, :2]) / A
                centroid_delta = (self.iot_positions[own_ids, :2].mean(axis=0) - self.positions[k, :2]) / A
                battery_mean = np.mean(self.batteries[own_ids]) / self.energy.battery_capacity_j
                battery_min = np.min(self.batteries[own_ids]) / self.energy.battery_capacity_j
            else:
                nearest_delta = np.zeros(2)
                centroid_delta = np.zeros(2)
                battery_mean = battery_min = 0.0
            features = np.r_[nearest_delta, centroid_delta, battery_mean, battery_min,
                             len(own_ids) / max(self.cfg.num_iot, 1), self.previous_collision[k]]
            vec = np.r_[own, capabilities, others, self.ris_phases[sl] / (2*np.pi),
                         self.previous_feedback[k], self.t / self.episode_steps, features]
            if vec.shape != (self.obs_dim,):
                raise RuntimeError(f'Observation shape mismatch: {vec.shape}, expected {self.obs_dim}')
            out.append(vec)
        return np.asarray(out, dtype=np.float32)

    def _state(self):
        # Variable-size global critic state; may be used by a CTDE MAPPO trainer.
        return np.asarray(np.r_[self.positions.flatten() / self.cfg.area_size_m,
                        self.velocities.flatten() / self.cfg.max_uav_speed_mps,
                        self.iot_positions.flatten() / self.cfg.area_size_m,
                        self.iot_active.astype(float), self.batteries / self.energy.battery_capacity_j,
                        self.eaves_positions.flatten() / self.cfg.area_size_m,
                        self.ris_phases / (2*np.pi), self.previous_feedback.flatten(),
                        self.t / self.episode_steps], dtype=np.float32)

    def step(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.n_agents, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f'Actions must be finite, shape ({self.n_agents}, {self.action_dim})')
        if not np.all((actions >= -1) & (actions <= 1)):
            raise ValueError('Actions must lie in [-1,1]')
        K, N, J = self.n_agents, self.cfg.num_iot, self.cfg.num_eavesdroppers
        dt, A = self.cfg.slot_duration_s, self.cfg.area_size_m
        # Refresh fading only at the start of each environment step. All repeated
        # channel evaluations in this step share the same reciprocal link samples.
        self.channel_model.begin_slot()
        previous_batteries = self.batteries.copy()
        previous_vel = self.velocities.copy()
        # Joint planar vector is normalized to ensure ||v|| <= vmax.
        direction = actions[:, :2].copy()
        norms = np.linalg.norm(direction, axis=1)
        direction /= np.maximum(1., norms)[:, None]
        attempted_displacement = direction * self.cfg.max_uav_speed_mps * dt
        candidate = self.positions.copy()
        candidate[:, :2] = np.clip(candidate[:, :2] + direction * self.cfg.max_uav_speed_mps * dt, 0, A)
        # Sequential safety filter. Check minimum separation throughout each
        # time slot (not merely end-point separation). Rejected UAVs hover.
        # Earlier accepted movements are also checked against later stationary
        # positions, so later rejections cannot create new conflicts.
        conflict = np.zeros(K, dtype=bool)
        starts = self.positions.copy()
        accepted = starts.copy()

        def swept_separation_ok(a0, a1, b0, b1):
            relative_start = a0 - b0
            relative_velocity = (a1 - a0) - (b1 - b0)
            denom = float(np.dot(relative_velocity, relative_velocity))
            alpha = np.clip(-np.dot(relative_start, relative_velocity) / denom, 0., 1.) if denom > 1e-15 else 0.
            minimum_gap = np.linalg.norm(relative_start + alpha * relative_velocity)
            return minimum_gap >= self.cfg.min_uav_separation_m - 1e-9

        for k in range(K):
            valid = True
            for m in range(K):
                if m == k:
                    continue
                other_end = accepted[m] if m < k else starts[m]
                if not swept_separation_ok(starts[k], candidate[k], starts[m], other_end):
                    valid = False
                    break
            if valid:
                accepted[k] = candidate[k]
            else:
                conflict[k] = True
        candidate = accepted
        self.previous_collision = conflict.astype(float)
        actual_displacement = candidate[:, :2] - self.positions[:, :2]
        actual_distance_by_uav = np.linalg.norm(actual_displacement, axis=1)
        attempted_distance_by_uav = np.linalg.norm(attempted_displacement, axis=1)
        self.velocities = actual_displacement / dt
        self.positions = candidate
        harvest_fracs = (actions[:, 3] + 1.) / 2.
        an_ratios = (actions[:, 4] + 1.) / 2. * self.max_an_ratio
        for k in range(K):
            sl = slice(k * self.ris_elements_per_agent, (k+1) * self.ris_elements_per_agent)
            self.ris_phases[sl] = np.pi * (actions[k, 5:] + 1.)
        # Bounded Gauss-like random walks are a simplified dynamic topology.
        self.iot_positions[:, :2] = np.clip(self.iot_positions[:, :2] + self.rng.normal(0, 0.4, (N, 2)), 0, A)
        self.eaves_positions[:, :2] = np.clip(self.eaves_positions[:, :2] + self.rng.normal(0, 0.2, (J, 2)), 0, A)
        self.iot_active = self.rng.random(N) < 0.9
        closest = np.argmin(np.linalg.norm(self.iot_positions[:, None, :2] - self.positions[None, :, :2], axis=2), axis=1)
        secrecy = np.zeros(N)
        legit = np.zeros(N)
        eaves = np.zeros(N)
        harvested = np.zeros(N)
        transmit_energy = np.zeros(N)
        charging_deficit = np.zeros(N)
        associated_count = np.bincount(closest[self.iot_active], minlength=K)
        per_uav_energy = np.zeros(K)
        for k in range(K):
            accel = (self.velocities[k] - previous_vel[k]) / dt
            per_uav_energy[k] = uav_slot_energy_j(self.velocities[k], harvest_fracs[k], an_ratios[k], dt,
                                                   acceleration_mps2=accel, params=self.energy)['total_j']
        for i in range(N):
            if not self.iot_active[i]:
                continue
            k = int(closest[i])
            # Use same phase vector per time step, separate fading per link.
            g_legit = self.channel_model.channel_gain(self.iot_positions[i], self.positions[k], self.ris_phases)
            harvested[i] = harvested_energy_j(g_legit, harvest_fracs[k], dt, self.energy)
            battery_result = iot_battery_step(self.batteries[i], harvested[i], harvest_fracs[k], dt,
                                               self.requested_iot_tx_w, self.energy)
            self.batteries[i] = battery_result['battery_j']
            transmit_energy[i] = battery_result['tx_energy_j'] + battery_result['circuit_energy_j']
            charging_deficit[i] = max(0.0, self.target_harvest_j_per_active_iot - harvested[i])
            tx = battery_result['tx_power_w']
            if tx <= 0:
                continue
            device_eaves = np.array([self.channel_model.channel_gain(self.iot_positions[i], e, self.ris_phases)
                                     for e in self.eaves_positions])
            uav_eaves = np.array([self.channel_model.channel_gain(self.positions[k], e, self.ris_phases)
                                  for e in self.eaves_positions])
            rate = secrecy_rate_bps(tx, g_legit, device_eaves, uav_eaves,
                                    an_ratios[k] * self.energy.uav_max_an_power_w,
                                    self.bandwidth_hz / max(int(associated_count[k]), 1), self.noise_power_w,
                                    harvest_fracs[k])
            legit[i] = rate['legitimate_rate_bps']
            eaves[i] = rate['eavesdropper_rate_bps']
            secrecy[i] = rate['secrecy_rate_bps']
        local_secrecy = np.array([np.mean(secrecy[(closest == k) & self.iot_active])
                                  if associated_count[k] else 0.0 for k in range(K)])
        # Feedback: secrecy Mbps, normalized energy, harvest fraction.
        self.previous_feedback = np.column_stack((local_secrecy / 1e6,
                                                   per_uav_energy / 150., harvest_fracs))
        # Reward = local normalized secrecy - energy cost; 50/50 local+global.
        local_reward = local_secrecy / 1e6 - 0.1 * per_uav_energy / 150.
        global_reward = np.mean(local_reward)
        reward = 0.5 * local_reward + 0.5 * global_reward
        # Optional safety and WPT-service penalties, all measured per UAV.
        # The target is a diagnostic SERVICE target, not a guarantee of RF feasibility.
        deficit_by_uav = np.bincount(closest[self.iot_active],
                                    weights=charging_deficit[self.iot_active], minlength=K)
        if self.target_harvest_j_per_active_iot > 0:
            deficit_fraction = deficit_by_uav / (np.maximum(associated_count, 1) *
                                                self.target_harvest_j_per_active_iot)
        else:
            deficit_fraction = np.zeros(K)
        local_penalty = self.collision_penalty * conflict.astype(float) + \
                        self.charging_deficit_penalty * deficit_fraction
        reward -= 0.5 * local_penalty + 0.5 * np.mean(local_penalty)
        self.t += 1
        done = self.t >= self.episode_steps
        info = {'secrecy_rate_bps': float(np.sum(secrecy)),
                'legitimate_rate_bps': float(np.sum(legit)),
                'eavesdropper_rate_bps': float(np.sum(eaves)),
                'mean_secrecy_rate_bps': float(np.mean(local_secrecy)),
                'harvested_energy_j': float(np.sum(harvested)),
                'uav_energy_j': float(np.sum(per_uav_energy)),
                'iot_batteries_j': self.batteries.copy(),
                'uav_positions_m': self.positions.copy(),
                'associations': closest.copy(),
                'active_iot_count': int(np.sum(self.iot_active)),
                'collision_reverted': int(np.sum(conflict)),
                'collision_reverted_fraction': float(np.mean(conflict)),
                'collision_attempt_by_uav': conflict.copy(),
                'attempted_distance_m': float(np.sum(attempted_distance_by_uav)),
                'actual_distance_m': float(np.sum(actual_distance_by_uav)),
                'mean_iot_battery_j': float(np.mean(self.batteries)),
                'min_iot_battery_j': float(np.min(self.batteries)),
                'iot_transmission_energy_j': float(np.sum(transmit_energy)),
                'net_iot_battery_change_j': float(np.sum(self.batteries - previous_batteries)),
                'active_iot_harvest_to_consumption_ratio': float(
                    np.sum(harvested) / max(np.sum(transmit_energy), 1e-15)),
                'active_iot_energy_neutral': bool(
                    np.sum(harvested) >= np.sum(transmit_energy) and np.sum(transmit_energy) > 0),
                'depleted_iot_fraction': float(np.mean(self.batteries <= 1e-6)),
                'charging_deficit_j': float(np.sum(charging_deficit)),
                'charging_deficit_fraction': float(np.mean(deficit_fraction)),
                'collision_penalty_total': float(np.sum(self.collision_penalty * conflict)),
                'max_uav_speed_mps': float(np.max(np.linalg.norm(self.velocities, axis=1)))}
        return self._observations(), self._state(), reward.astype(np.float32), done, info
