from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    # A small reproducible starting scenario (illustrative, not paper-calibrated).
    seed: int = 42
    num_uavs: int = 3
    num_iot: int = 12
    num_eavesdroppers: int = 2
    num_ris_elements: int = 32
    max_uavs: int = 8
    area_size_m: float = 200.0
    uav_altitude_m: float = 40.0
    ris_position: tuple = (100.0, 100.0, 20.0)
    slot_duration_s: float = 1.0
    max_uav_speed_mps: float = 10.0
    min_uav_separation_m: float = 5.0

    # Channel parameters: illustrative defaults to be calibrated later.
    reference_gain: float = 1e-3
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.2
    nlos_attenuation: float = 0.15
    los_a: float = 9.61
    los_b: float = 0.16
    rician_k: float = 3.0

    def __post_init__(self):
        if min(self.num_uavs, self.num_iot, self.num_ris_elements) < 1:
            raise ValueError('UAV, IoT, and RIS counts must be positive')
        if self.num_eavesdroppers < 0 or self.num_uavs > self.max_uavs:
            raise ValueError('Invalid eavesdropper count or maximum UAV count')
        if self.num_ris_elements % self.max_uavs != 0:
            raise ValueError('RIS element count must be divisible by max_uavs')
        if self.area_size_m <= 0 or self.reference_distance_m <= 0:
            raise ValueError('Distances must be positive')
