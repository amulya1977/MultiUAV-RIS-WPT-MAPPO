"""Run from project root: python test_security.py"""
import numpy as np
from pls_mappo.config import SimulationConfig
from pls_mappo.channel_model import RISChannelModel
from pls_mappo.security_model import secrecy_rate_bps


def main():
    cfg = SimulationConfig()
    channel = RISChannelModel(cfg)
    phases = np.zeros(cfg.num_ris_elements)
    uav = np.array([50., 60., cfg.uav_altitude_m])
    device = np.array([70., 65., 0.])
    eaves = [np.array([100., 80., 0.]), np.array([135., 125., 0.])]

    # Evaluate each link once and hold it fixed during the AN sweep.
    # This avoids confusing fading randomness with the effect of AN power.
    legit_gain = channel.channel_gain(device, uav, phases)
    dev_gains = np.array([channel.channel_gain(device, e, phases) for e in eaves])
    uav_gains = np.array([channel.channel_gain(uav, e, phases) for e in eaves])

    common = dict(tx_power_w=0.05, legitimate_gain=legit_gain,
                  device_to_eaves_gains=dev_gains,
                  uav_to_eaves_gains=uav_gains, bandwidth_hz=1e6,
                  noise_power_w=1e-12, harvest_fraction=0.4, snr_gap=1.5)

    results = []
    for an_power in [0.0, 0.1, 1.0]:
        r = secrecy_rate_bps(an_power_w=an_power, **common)
        results.append(r)
        print(f"AN = {an_power:.1f} W | legitimate = {r['legitimate_rate_bps']:.2f} bps | "
              f"eavesdropper = {r['eavesdropper_rate_bps']:.2f} bps | "
              f"secrecy = {r['secrecy_rate_bps']:.2f} bps")
        assert all(np.isfinite(v) and v >= 0 for v in r.values())

    assert np.allclose([r['legitimate_rate_bps'] for r in results],
                       results[0]['legitimate_rate_bps'])
    assert all(results[i+1]['eavesdropper_rate_bps'] <=
               results[i]['eavesdropper_rate_bps'] + 1e-7 for i in range(2))
    assert all(results[i+1]['secrecy_rate_bps'] + 1e-7 >=
               results[i]['secrecy_rate_bps'] for i in range(2))

    # Secrecy is clipped to zero if an eavesdropper has a stronger channel.
    weaker = secrecy_rate_bps(0.05, 1e-12, np.array([1e-3]),
                              np.array([0.]), 0., 1e6, 1e-12, 0.4)
    assert weaker['secrecy_rate_bps'] == 0.0

    # Without eavesdroppers, secrecy equals the legitimate rate.
    no_eaves = secrecy_rate_bps(0.05, legit_gain, [], [], 0., 1e6, 1e-12, 0.4)
    assert np.isclose(no_eaves['secrecy_rate_bps'], no_eaves['legitimate_rate_bps'])
    print('Security model test PASSED')


if __name__ == '__main__':
    main()
