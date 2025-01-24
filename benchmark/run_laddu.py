#!/usr/bin/env python3

from timeit import default_timer as timer
import pickle
from pathlib import Path

import laddu as ld
import numpy as np


def main():
    tmp = Path('tmp')
    tmp.mkdir(exist_ok=True)
    data_path = Path('data.root')
    accmc_path = Path('accmc.root')
    data = ld.open_amptools(str(data_path), pol_angle=0.0, pol_magnitude=0.3519)
    accmc = ld.open_amptools(str(accmc_path), pol_angle=0.0, pol_magnitude=0.3519)
    output = fit_unbinned(niters=50, data_ds=data, accmc_ds=accmc)
    pickle.dump(output, (tmp / 'laddu_fit.pkl').open('wb'))


def fit_unbinned(
    niters: int,
    data_ds: ld.Dataset,
    accmc_ds: ld.Dataset,
) -> tuple[ld.Status, list[ld.Status]]:
    res_mass = ld.Mass([2, 3])
    angles = ld.Angles(0, [1], [2], [2, 3])
    polarization = ld.Polarization(0, [1])
    manager = ld.Manager()
    z00p = manager.register(ld.Zlm('Z00+', 0, 0, '+', angles, polarization))
    z22p = manager.register(ld.Zlm('Z22+', 2, 2, '+', angles, polarization))
    bw_f01500 = manager.register(
        ld.BreitWigner(
            'f0(1500)',
            ld.constant(1.506),
            ld.parameter('f0_width'),
            0,
            ld.Mass([2]),
            ld.Mass([3]),
            res_mass,
        )
    )
    bw_f21525 = manager.register(
        ld.BreitWigner(
            'f2(1525)',
            ld.constant(1.517),
            ld.parameter('f2_width'),
            2,
            ld.Mass([2]),
            ld.Mass([3]),
            res_mass,
        )
    )
    s0p = manager.register(ld.Scalar('S0+', ld.parameter('S0+ re')))
    d2p = manager.register(ld.ComplexScalar('D2+', ld.parameter('D2+ re'), ld.parameter('D2+ im')))
    pos_re = (s0p * bw_f01500 * z00p.real() + d2p * bw_f21525 * z22p.real()).norm_sqr()
    pos_im = (s0p * bw_f01500 * z00p.imag() + d2p * bw_f21525 * z22p.imag()).norm_sqr()
    model = manager.model(pos_re + pos_im)

    rng = np.random.default_rng(0)

    best_nll = np.inf
    best_status = None
    all_statuses = []
    bounds = [
        (0.001, 1.0),
        (0.001, 1.0),
        (None, None),
        (None, None),
        (None, None),
    ]
    nll = ld.NLL(model, data_ds, accmc_ds)
    for iiter in range(niters):
        print(f'Fitting iteration {iiter}/50')
        start = timer()
        p0 = rng.uniform(-2000.0, 2000.0, 3)
        p0 = np.append([0.8, 0.5], p0)
        status = nll.minimize(p0, bounds=bounds, threads=1)
        all_statuses.append(status)
        if status.fx < best_nll:
            best_nll = status.fx
            best_status = status
            print('=== NEW BEST ===')
        end = timer()
        print(status)
        print(f'Time elapsed: {end - start}s')
    return best_status, all_statuses


if __name__ == '__main__':
    main()
