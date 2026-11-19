"""Stage 3: WPT, finite IoT battery, and rotary-wing UAV energy accounting.

Units: power W, energy J, time s, positions m, velocities m/s.
The physical parameters are illustrative, not paper-calibrated.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EnergyParameters:
    battery_capacity_j: float = 2.0
    initial_battery_j: float = 0.5
    rf_to_dc_efficiency: float = 0.6
    wpt_power_w: float = 10.0
    iot_max_tx_power_w: float = 0.1
    iot_circuit_power_w: float = 0.005
    uav_max_an_power_w: float = 1.0
    blade_profile_power_w: float = 79.86
    induced_power_w: float = 88.63
    rotor_tip_speed_mps: float = 120.0
    induced_velocity_mps: float = 4.03
    fuselage_drag_ratio: float = 0.6
    air_density_kg_m3: float = 1.225
    rotor_solidity: float = 0.05
    rotor_disk_area_m2: float = 0.503
    uav_mass_kg: float = 1.5
    acceleration_penalty: float = 0.05
    hovering_speed_threshold_mps: float = 0.01

    def __post_init__(self):
        positive = (self.battery_capacity_j, self.wpt_power_w,
                    self.rotor_tip_speed_mps, self.induced_velocity_mps,
                    self.rotor_disk_area_m2)
        if any(v <= 0 or not np.isfinite(v) for v in positive):
            raise ValueError('Required physical parameters must be positive and finite')
        if not 0 <= self.initial_battery_j <= self.battery_capacity_j:
            raise ValueError('Initial battery must be between zero and capacity')
        if not 0 < self.rf_to_dc_efficiency <= 1:
            raise ValueError('RF-to-DC efficiency must be in (0,1]')


def _bounded(name, value, lo, hi):
    value = float(value)
    if not np.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f'{name} must be finite and in [{lo}, {hi}]')
    return value


def harvested_energy_j(channel_gain, harvest_fraction, slot_s, params=EnergyParameters()):
    """Eq. (9): eta * P_wpt * |h|^2 * tau_h * slot_length."""
    gain = float(channel_gain)
    if not np.isfinite(gain) or gain < 0:
        raise ValueError('Channel power gain must be nonnegative and finite')
    harvest_fraction = _bounded('harvest_fraction', harvest_fraction, 0, 1)
    if slot_s <= 0 or not np.isfinite(slot_s):
        raise ValueError('slot_s must be positive and finite')
    return params.rf_to_dc_efficiency * params.wpt_power_w * gain * harvest_fraction * slot_s


def iot_battery_step(battery_j, harvested_j, harvest_fraction, slot_s,
                     requested_tx_power_w, params=EnergyParameters()):
    """Harvest-then-transmit within a slot. Returns battery, TX power and energy use.

    Unlike the paper's Eq. (10), which caps TX by pre-harvest battery, this
    allows same-slot harvested RF energy to be used in the information phase.
    Circuit power is charged only when transmission is possible.
    """
    battery = _bounded('battery_j', battery_j, 0, params.battery_capacity_j)
    harvest = float(harvested_j)
    if not np.isfinite(harvest) or harvest < 0:
        raise ValueError('harvested_j must be finite and nonnegative')
    fraction = _bounded('harvest_fraction', harvest_fraction, 0, 1)
    requested = _bounded('requested_tx_power_w', requested_tx_power_w, 0, params.iot_max_tx_power_w)
    if slot_s <= 0 or not np.isfinite(slot_s):
        raise ValueError('slot_s must be positive and finite')
    available = min(params.battery_capacity_j, battery + harvest)
    info_time = (1.0 - fraction) * slot_s
    if info_time <= 0 or available <= params.iot_circuit_power_w * info_time:
        return {'battery_j': available, 'tx_power_w': 0.0, 'tx_energy_j': 0.0,
                'circuit_energy_j': 0.0, 'harvested_j': harvest}
    max_tx = max(0.0, available / info_time - params.iot_circuit_power_w)
    actual_power = min(requested, max_tx)
    circuit_energy = params.iot_circuit_power_w * info_time if actual_power > 0 else 0.0
    tx_energy = actual_power * info_time
    return {'battery_j': max(0.0, available - tx_energy - circuit_energy),
            'tx_power_w': actual_power, 'tx_energy_j': tx_energy,
            'circuit_energy_j': circuit_energy, 'harvested_j': harvest}


def uav_propulsion_power_w(velocity_mps, acceleration_mps2=None, params=EnergyParameters()):
    """Standard rotary-wing induced + blade profile + parasite drag model.

    Uses positive induced power, unlike the apparent negative sign in the
    extracted paper Eq. (11). Clamps maneuvering contribution to avoid
    negative net propulsion power during deceleration.
    """
    vel = np.asarray(velocity_mps, dtype=float)
    acc = np.zeros_like(vel) if acceleration_mps2 is None else np.asarray(acceleration_mps2, dtype=float)
    if vel.shape != acc.shape or vel.ndim != 1 or vel.size not in (2, 3):
        raise ValueError('Velocity and acceleration must be matching 2D/3D vectors')
    if not np.isfinite(vel).all() or not np.isfinite(acc).all():
        raise ValueError('Velocity and acceleration must be finite')
    v = float(np.linalg.norm(vel))
    p = params
    if v <= p.hovering_speed_threshold_mps:
        return p.blade_profile_power_w + p.induced_power_w
    blade = p.blade_profile_power_w * (1 + 3 * v**2 / p.rotor_tip_speed_mps**2)
    induced_factor = np.sqrt(np.sqrt(1 + v**4 / (4 * p.induced_velocity_mps**4)) -
                             v**2 / (2 * p.induced_velocity_mps**2))
    induced = p.induced_power_w * induced_factor
    drag = 0.5 * p.fuselage_drag_ratio * p.air_density_kg_m3 * p.rotor_solidity * p.rotor_disk_area_m2 * v**3
    maneuver = p.uav_mass_kg * float(np.dot(vel, acc)) + p.acceleration_penalty * float(np.dot(acc, acc))
    return max(0.0, blade + induced + drag + maneuver)


def uav_slot_energy_j(velocity_mps, harvest_fraction, an_ratio, slot_s,
                      acceleration_mps2=None, params=EnergyParameters()):
    """Eq. (15) energy accounting: propulsion + WPT broadcast + AN."""
    fraction = _bounded('harvest_fraction', harvest_fraction, 0, 1)
    ratio = _bounded('an_ratio', an_ratio, 0, 1)
    if slot_s <= 0 or not np.isfinite(slot_s):
        raise ValueError('slot_s must be positive and finite')
    propulsion = uav_propulsion_power_w(velocity_mps, acceleration_mps2, params) * slot_s
    wpt = params.wpt_power_w * fraction * slot_s
    an = params.uav_max_an_power_w * ratio * (1 - fraction) * slot_s
    return {'propulsion_j': propulsion, 'wpt_j': wpt,
            'artificial_noise_j': an, 'total_j': propulsion + wpt + an}
