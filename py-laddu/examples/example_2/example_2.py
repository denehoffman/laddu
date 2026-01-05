#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "docopt-ng",
#     "laddu",
#     "matplotlib",
#     "numpy",
# ]
# ///
"""
Usage: example_2.py [-n <nbins>] [-l <lmax>] [-b <nboot>]

Options:
-n <nbins>   Number of bins to use in binned fit and plot [default: 40]
-l <lmax>    Maximum L to use in analysis [default: 4]
-b <nboot>   Number of bootstrapped fits to perform for each fit [default: 10]
"""  # noqa: D400

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

import laddu as ld

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_measured_moment(data: ld.Dataset, *, i: int, l: int, m: int) -> complex:
    const = 2 * np.sqrt((4 * np.pi) / (2 * l + 1)) * (1 / 2 if i == 0 else 1)
    topology = ld.Topology.missing_k2('beam', ['kshort1', 'kshort2'], 'proton')
    polarization = ld.Polarization(
        topology, pol_magnitude='pol_magnitude', pol_angle='pol_angle'
    )
    big_phi = data[polarization.pol_angle]
    p_gamma = data[polarization.pol_magnitude]
    pol_term = np.ones(data.n_events)
    if i == 1:
        pol_term = np.cos(2 * big_phi) / p_gamma
    elif i == 2:
        pol_term = np.sin(2 * big_phi) / p_gamma
    ylm = ld.Ylm(
        'ylm',
        l,
        m,
        ld.Angles(
            ld.Topology.missing_k2('beam', ['kshort1', 'kshort2'], 'proton'), 'kshort1'
        ),
    )
    model = ylm.conj()
    evaluator = model.load(data)
    values = evaluator.evaluate([])
    weights = data.weights
    return const * np.sum(weights * values * pol_term)


def get_norm_int_term(
    accmc: ld.Dataset,
    *,
    n_gen: int,
    i: int,
    l: int,
    m: int,
    ip: int,
    lp: int,
    mp: int,
) -> complex:
    const = (
        8.0 * np.pi / n_gen * np.sqrt((2 * lp + 1) / (2 * l + 1)) * (1j if ip == 2 else 1)
    )
    topology = ld.Topology.missing_k2('beam', ['kshort1', 'kshort2'], 'proton')
    polarization = ld.Polarization(
        topology, pol_magnitude='pol_magnitude', pol_angle='pol_angle'
    )
    big_phi = accmc[polarization.pol_angle]
    p_gamma = accmc[polarization.pol_magnitude]
    pol_term = np.ones(accmc.n_events)
    if i == 1:
        pol_term = np.cos(2 * big_phi) / p_gamma
    elif i == 2:
        pol_term = np.sin(2 * big_phi) / p_gamma
    if ip == 1:
        pol_term *= np.cos(2 * big_phi) * p_gamma
    elif ip == 2:
        pol_term *= np.sin(2 * big_phi) * p_gamma
    ylm = ld.Ylm(
        'ylm',
        l,
        m,
        ld.Angles(
            ld.Topology.missing_k2('beam', ['kshort1', 'kshort2'], 'proton'), 'kshort1'
        ),
    )
    ylpmp = ld.Ylm(
        'ylpmp',
        lp,
        mp,
        ld.Angles(
            ld.Topology.missing_k2('beam', ['kshort1', 'kshort2'], 'proton'), 'kshort1'
        ),
    )
    model = ylm.conj() * (ylpmp.imag() if ip == 2 else ylpmp.real())
    evaluator = model.load(accmc)
    values = evaluator.evaluate([])
    weights = accmc.weights
    val = const * np.sum(weights * values * pol_term)
    return val


def get_dim(*, l_max: int, polarized: bool) -> int:
    return len(
        [
            None
            for i in range(3 if polarized else 1)
            for l in range(l_max + 1)
            for m in range(l + 1)
            if not (i == 2 and m == 0)
        ]
    )


def get_ilms(*, l_max: int, polarized: bool) -> list[tuple[int, int, int]]:
    return [
        (i, l, m)
        for i in range(3 if polarized else 1)
        for l in range(l_max + 1)
        for m in range(l + 1)
        if not (i == 2 and m == 0)
    ]


def calculate_moments(
    data: ld.Dataset, accmc: ld.Dataset, *, n_gen: int, l_max: int, polarized: bool
) -> NDArray[np.complex128]:
    dim = get_dim(l_max=l_max, polarized=polarized)
    measured_moments = np.zeros(dim, dtype=np.complex128)
    for j, (i, l, m) in enumerate(get_ilms(l_max=l_max, polarized=polarized)):
        measured_moments[j] = get_measured_moment(data, i=i, l=l, m=m)
    norm_int_matrix = np.zeros((dim, dim), dtype=np.complex128)
    for j, (i, l, m) in enumerate(get_ilms(l_max=l_max, polarized=polarized)):
        for k, (ip, lp, mp) in enumerate(get_ilms(l_max=l_max, polarized=polarized)):
            norm_int_matrix[j, k] = get_norm_int_term(
                accmc,
                n_gen=n_gen,
                i=i,
                l=l,
                m=m,
                ip=ip,
                lp=lp,
                mp=mp,
            )
    norm_int_matrix_inv = np.linalg.inv(norm_int_matrix)
    return np.dot(norm_int_matrix_inv, measured_moments)


def get_names(*, l_max: int, polarized: bool) -> list[tuple[str, int, int, int]]:
    return [
        (
            (r'$\Im[' if i == 2 else '$')
            + f'H_{i}({l}, {m})'
            + (']$' if i == 2 else '$'),
            i,
            l,
            m,
        )
        for i in range(3 if polarized else 1)
        for l in range(l_max + 1)
        for m in range(l + 1)
        if not (i == 2 and m == 0)
    ]


if __name__ == '__main__':
    args = docopt(__doc__ or '')
    script_dir = Path(os.path.realpath(__file__)).parent.resolve()
    data_dir = script_dir.parent / 'data'
    data_file = data_dir / 'data.parquet'
    accmc_file = data_dir / 'accmc.parquet'
    bins = int(args['-n'])
    nboot = int(args['-b'])
    l_max = int(args['-l'])
    edges = np.histogram_bin_edges([], bins, (1.0, 2.0))
    centers = (edges[1:] + edges[:-1]) / 2

    p4_columns = ['beam', 'proton', 'kshort1', 'kshort2']
    aux_columns = ['pol_magnitude', 'pol_angle']

    if not (script_dir / 'unpolarized_moments.pkl').exists():
        data = ld.io.read_parquet(
            data_file,
            p4s=p4_columns,
            aux=aux_columns,
        )
        accmc = ld.io.read_parquet(
            accmc_file,
            p4s=p4_columns,
            aux=aux_columns,
        )
        mass = ld.Mass(['kshort1', 'kshort2'])
        data_binned = data.bin_by(mass, bins, (1.0, 2.0))
        accmc_binned = accmc.bin_by(mass, bins, (1.0, 2.0))
        binned_moments = []
        bootstrapped_moments = []
        for ibin in range(bins):
            print(f'Calculating moments for bin {ibin}')
            n_gen = accmc_binned[ibin].n_events * 100
            moments = calculate_moments(
                data_binned[ibin],
                accmc_binned[ibin],
                n_gen=n_gen,
                l_max=l_max,
                polarized=False,
            )
            binned_moments.append(moments)
            b_moments = []
            for iboot in range(nboot):
                print(f'Bootstrapping bin {ibin}, boot {iboot}')
                b_i_moments = calculate_moments(
                    data_binned[ibin].bootstrap(iboot),
                    accmc_binned[ibin],
                    n_gen=n_gen,
                    l_max=l_max,
                    polarized=False,
                )
                b_moments.append(b_i_moments)
            bootstrapped_moments.append(b_moments)
        pickle.dump(
            {'moments': binned_moments, 'bootstraps': bootstrapped_moments},
            (script_dir / 'unpolarized_moments.pkl').open('wb'),
        )
    unpolarized_moments = pickle.load((script_dir / 'unpolarized_moments.pkl').open('rb'))
    _, ax = plt.subplots(l_max + 1, l_max + 1, figsize=(3 * (l_max + 1), 3 * (l_max + 1)))
    for imoment, (name, _, l, m) in enumerate(get_names(l_max=l_max, polarized=False)):
        ax[l, m].errorbar(
            centers,
            [moments[imoment].real for moments in unpolarized_moments['moments']],
            yerr=[
                np.std([b_moments[imoment].real for b_moments in moments], ddof=1)
                for moments in unpolarized_moments['bootstraps']
            ],
            fmt='.',
        )
        ax[l, m].set_title(name)
        ax[l, m].set_xlabel('Mass of $K_S^0 K_S^0$ (GeV/$c^2$)')
    for l in range(l_max + 1):
        for m in range(l_max + 1):
            if m > l:
                ax[l, m].set_visible(False)
    plt.tight_layout()
    plt.savefig('moments.svg')
    plt.close()

    if not (script_dir / 'polarized_moments.pkl').exists():
        data = ld.io.read_parquet(data_file, p4s=p4_columns, aux=aux_columns)
        accmc = ld.io.read_parquet(accmc_file, p4s=p4_columns, aux=aux_columns)
        mass = ld.Mass(['kshort1', 'kshort2'])
        data_binned = data.bin_by(mass, bins, (1.0, 2.0))
        accmc_binned = accmc.bin_by(mass, bins, (1.0, 2.0))
        binned_moments = []
        bootstrapped_moments = []
        for ibin in range(bins):
            print(f'Calculating polarized moments for bin {ibin}')
            n_gen = accmc_binned[ibin].n_events * 100
            moments = calculate_moments(
                data_binned[ibin],
                accmc_binned[ibin],
                n_gen=n_gen,
                l_max=l_max,
                polarized=True,
            )
            binned_moments.append(moments)
            b_moments = []
            for iboot in range(nboot):
                print(f'Bootstrapping bin {ibin}, boot {iboot}')
                b_i_moments = calculate_moments(
                    data_binned[ibin].bootstrap(iboot),
                    accmc_binned[ibin],
                    n_gen=n_gen,
                    l_max=l_max,
                    polarized=True,
                )
                b_moments.append(b_i_moments)
            bootstrapped_moments.append(b_moments)
        pickle.dump(
            {'moments': binned_moments, 'bootstraps': bootstrapped_moments},
            (script_dir / 'polarized_moments.pkl').open('wb'),
        )
    polarized_moments = pickle.load((script_dir / 'polarized_moments.pkl').open('rb'))
    for j in range(3):
        _, ax = plt.subplots(
            l_max + (1 if j != 2 else 0),
            l_max + (1 if j != 2 else 0),
            figsize=(
                3 * (l_max + (1 if j != 2 else 0)),
                3 * (l_max + (1 if j != 2 else 0)),
            ),
        )
        for imoment, (name, i, l, m) in enumerate(get_names(l_max=l_max, polarized=True)):
            if i != j:
                continue
            ax[l - (1 if j == 2 else 0), m - (1 if j == 2 else 0)].errorbar(
                centers,
                [moments[imoment].real for moments in polarized_moments['moments']],
                yerr=[
                    np.std([b_moments[imoment].real for b_moments in moments], ddof=1)
                    for moments in polarized_moments['bootstraps']
                ],
                fmt='.',
            )
            ax[l - (1 if j == 2 else 0), m - (1 if j == 2 else 0)].set_title(name)
            ax[l - (1 if j == 2 else 0), m - (1 if j == 2 else 0)].set_xlabel(
                'Mass of $K_S^0 K_S^0$ (GeV/$c^2$)'
            )
        for l in range(l_max + (1 if j != 2 else 0)):
            for m in range(l_max + (1 if j != 2 else 0)):
                if m > l:
                    ax[l, m].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'polarized_moments_{j}.svg')
        plt.close()
