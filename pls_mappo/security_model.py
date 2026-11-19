"""Physical-layer security model based on Group8 equations (6)--(8).

Rates are effective bit/s, including the information-phase time fraction.
Inputs are channel *power* gains |h|^2, not complex amplitudes.
Noise is total receiver noise power in watts, not noise spectral density.
The serving UAV is assumed to cancel its own artificial noise perfectly.
"""

import numpy as np


def _nonnegative(name, x):
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError(f'{name} must be finite and nonnegative')
    return x


def _fraction(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('harvest_fraction must lie in [0, 1]')
    return value


def legitimate_rate_bps(tx_power_w, legitimate_gain, bandwidth_hz,
                        noise_power_w, harvest_fraction, snr_gap=1.0):
    """Eq. (6): (1 - tau_h) B log2(1 + P_i |h_ki|^2 / (eta*sigma^2))."""
    p = float(_nonnegative('tx_power_w', tx_power_w))
    gain = float(_nonnegative('legitimate_gain', legitimate_gain))
    bw = float(_nonnegative('bandwidth_hz', bandwidth_hz))
    noise = float(noise_power_w)
    gap = float(snr_gap)
    t = _fraction(harvest_fraction)
    if not np.isfinite(noise) or noise <= 0 or not np.isfinite(gap) or gap < 1:
        raise ValueError('noise_power_w must be positive, snr_gap must be >= 1')
    return float((1 - t) * bw * np.log2(1 + p * gain / (gap * noise)))


def worst_eavesdropper_rate_bps(tx_power_w, device_to_eaves_gains,
                                 uav_to_eaves_gains, an_power_w,
                                 bandwidth_hz, noise_power_w,
                                 harvest_fraction, snr_gap=1.0):
    """Eq. (7): worst-case rate across eavesdroppers under UAV artificial noise.

    Device and UAV gains must be equal-length vectors; empty vectors give zero
    eavesdropper rate. Both gains should be computed with consistent RIS phases.
    """
    p = float(_nonnegative('tx_power_w', tx_power_w))
    g_dev = _nonnegative('device_to_eaves_gains', device_to_eaves_gains)
    g_uav = _nonnegative('uav_to_eaves_gains', uav_to_eaves_gains)
    an = float(_nonnegative('an_power_w', an_power_w))
    bw = float(_nonnegative('bandwidth_hz', bandwidth_hz))
    noise = float(noise_power_w)
    gap = float(snr_gap)
    t = _fraction(harvest_fraction)
    if g_dev.ndim != 1 or g_uav.shape != g_dev.shape:
        raise ValueError('Eavesdropper gain arrays must be 1-D and equal-length')
    if not np.isfinite(noise) or noise <= 0 or not np.isfinite(gap) or gap < 1:
        raise ValueError('noise_power_w must be positive, snr_gap must be >= 1')
    if not g_dev.size:
        return 0.0
    sinr = p * g_dev / (an * g_uav + gap * noise)
    return float(np.max((1 - t) * bw * np.log2(1 + sinr)))


def secrecy_rate_bps(tx_power_w, legitimate_gain, device_to_eaves_gains,
                     uav_to_eaves_gains, an_power_w, bandwidth_hz,
                     noise_power_w, harvest_fraction, snr_gap=1.0):
    """Eq. (8): [legitimate rate - worst-case eavesdropper rate]^+."""
    legitimate = legitimate_rate_bps(tx_power_w, legitimate_gain, bandwidth_hz,
                                     noise_power_w, harvest_fraction, snr_gap)
    eaves = worst_eavesdropper_rate_bps(tx_power_w, device_to_eaves_gains,
                                       uav_to_eaves_gains, an_power_w,
                                       bandwidth_hz, noise_power_w,
                                       harvest_fraction, snr_gap)
    return {
        'legitimate_rate_bps': legitimate,
        'eavesdropper_rate_bps': eaves,
        'secrecy_rate_bps': max(0.0, legitimate - eaves),
    }
