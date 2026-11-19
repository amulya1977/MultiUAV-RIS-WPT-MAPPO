import numpy as np
from .config import SimulationConfig


class RISChannelModel:
    """Direct + RIS-cascaded complex channel, corresponding to paper Eq. (3).

    Small-scale fading is sampled once per link per evaluation. The geometry-based
    path loss and probabilistic LoS model are simplified research prototypes.
    """

    def __init__(self, config: SimulationConfig, seed=None):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed if seed is None else seed)
        # One geometry-specific fading realization per link in a simulation slot.
        # Calling begin_slot() replaces all stored fading, without changing RNG seed.
        self._slot_cache = {}
        self.slot_index = 0

    def begin_slot(self):
        self.slot_index += 1
        self._slot_cache.clear()

    @staticmethod
    def _link_key(source, destination, count):
        # Reciprocal TDD link: A->B and B->A refer to the same sampled coefficients.
        a = tuple(np.asarray(source, dtype=float).tolist())
        b = tuple(np.asarray(destination, dtype=float).tolist())
        return (a, b, count) if a <= b else (b, a, count)

    @staticmethod
    def _point(point):
        p = np.asarray(point, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('Each position must be a finite 3-vector (x,y,z)')
        return p

    def _los_probability(self, source, destination):
        dz = abs(float(source[2] - destination[2]))
        d = max(float(np.linalg.norm(source - destination)), 1e-9)
        elevation_deg = np.degrees(np.arcsin(np.clip(dz / d, 0.0, 1.0)))
        return 1.0 / (1.0 + self.cfg.los_a * np.exp(-self.cfg.los_b * (elevation_deg - self.cfg.los_a)))

    def _link_vector(self, source, destination, count=1):
        source, destination = self._point(source), self._point(destination)
        key = self._link_key(source, destination, count)
        if key in self._slot_cache:
            return self._slot_cache[key].copy()
        d = max(float(np.linalg.norm(source - destination)), self.cfg.reference_distance_m)
        los = self.rng.random() < self._los_probability(source, destination)
        scale = self.cfg.reference_gain * (d / self.cfg.reference_distance_m) ** (-self.cfg.path_loss_exponent)
        if not los:
            scale *= self.cfg.nlos_attenuation
        rayleigh = (self.rng.normal(size=count) + 1j * self.rng.normal(size=count)) / np.sqrt(2)
        if los:
            k = self.cfg.rician_k
            fading = np.sqrt(k / (k + 1)) + np.sqrt(1 / (k + 1)) * rayleigh
        else:
            fading = rayleigh
        link = np.sqrt(scale) * fading
        self._slot_cache[key] = link.copy()
        return link

    def channel_components(self, source_position, destination_position):
        """Return fixed-in-slot direct and cascaded element coefficients.

        For physically identical endpoints, fading is reused within a slot.
        Changing RIS phases does not resample the channel.
        """
        ris = np.asarray(self.cfg.ris_position, dtype=float)
        direct = self._link_vector(source_position, destination_position)[0]
        sr = self._link_vector(source_position, ris, self.cfg.num_ris_elements)
        rd = self._link_vector(ris, destination_position, self.cfg.num_ris_elements)
        return complex(direct), sr * rd

    def channel(self, source_position, destination_position, phase_shifts):
        """Composite channel h = h_direct + sum(h_RtoDest * exp(j theta) * h_SourceToR)."""
        phase_shifts = np.asarray(phase_shifts, dtype=float)
        if phase_shifts.shape != (self.cfg.num_ris_elements,):
            raise ValueError(f'Expected {self.cfg.num_ris_elements} RIS phase shifts')
        if not np.isfinite(phase_shifts).all():
            raise ValueError('RIS phases must be finite')
        hd, cascaded_elements = self.channel_components(source_position, destination_position)
        return complex(hd + np.sum(cascaded_elements * np.exp(1j * phase_shifts)))

    def channel_gain(self, source_position, destination_position, phase_shifts):
        return float(abs(self.channel(source_position, destination_position, phase_shifts)) ** 2)
