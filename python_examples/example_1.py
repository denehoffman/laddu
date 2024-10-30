#!/usr/bin/env python3
"""
Usage: example_1.py [-n <nbins>]

Options:
-n <nbins>  Number of bins to use in binned fit and plot [default: 50]
"""

import os
from pathlib import Path
from time import perf_counter

import laddu as ld
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.umath as unp
from docopt import docopt
from rich import print as pprint
from rich.table import Table
from uncertainties import ufloat


def main(bins: int):  # noqa: PLR0915
    script_dir = Path(os.path.realpath(__file__)).parent.resolve()
    data_file = str(script_dir / "data_1.parquet")
    mc_file = str(script_dir / "mc_1.parquet")
    start = perf_counter()
    binned_tot_res, binned_s0p_res, binned_d2p_res, bin_edges = fit_binned(bins, data_file, mc_file)
    tot_weights, s0p_weights, d2p_weights, status, parameters = fit_unbinned(data_file, mc_file)
    end = perf_counter()
    pprint(f"Total time: {end - start:.3f}s")
    print(status)  # noqa: T201

    f0_width = status.x[parameters.index("f0_width")]
    f2_width = status.x[parameters.index("f2_width")]
    f0_re = status.x[parameters.index("S0+ re")]
    f2_re = status.x[parameters.index("D2+ re")]
    f2_im = status.x[parameters.index("D2+ im")]
    if status.err is not None:
        f0_width_err = status.err[parameters.index("f0_width")]
        f2_width_err = status.err[parameters.index("f2_width")]
        f0_re_err = status.err[parameters.index("S0+ re")]
        f2_re_err = status.err[parameters.index("D2+ re")]
        f2_im_err = status.err[parameters.index("D2+ im")]
    else:
        f0_width_err = np.inf
        f2_width_err = np.inf
        f0_re_err = np.inf
        f2_re_err = np.inf
        f2_im_err = np.inf

    u_f0_re = ufloat(f0_re, f0_re_err)
    u_f2_re = ufloat(f2_re, f2_re_err)
    u_f2_im = ufloat(f2_im, f2_im_err)
    sd_ratio = u_f0_re / unp.sqrt(unp.pow(u_f2_re, 2) + unp.pow(u_f2_im, 2))  # pyright: ignore
    sd_phase = unp.atan2(u_f2_im, u_f2_re)  # pyright: ignore

    table = Table("", "f0(1500) Width", "f2'(1525) Width", "S/D Ratio", "S-D Phase")
    table.add_row("Truth", "0.11200", "0.08600", f"{100 / np.sqrt(50**2 + 50**2):.5f}", f"{np.atan2(50, 50):.5f}")
    table.add_row(
        "Fit",
        f"{f0_width:.5f} ± {f0_width_err:.5f}",
        f"{f2_width:.5f} ± {f2_width_err:.5f}",
        f"{sd_ratio.n:.5f} ± {sd_ratio.s:.5f}",
        f"{sd_phase.n:.5f} ± {sd_phase.s:.5f}",
    )
    pprint(table)

    res_mass = ld.Mass([2, 3])
    data_ds = ld.open("./data_1.parquet")
    accmc_ds = ld.open("./mc_1.parquet")

    m_data = res_mass.value_on(data_ds)
    m_accmc = res_mass.value_on(accmc_ds)

    font = {"family": "DejaVu Sans", "weight": "normal", "size": 22}
    plt.rc("font", **font)
    plt.rc("axes", titlesize=48)
    plt.rc("legend", fontsize=24)
    plt.rc("xtick", direction="in")
    plt.rc("ytick", direction="in")
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["xtick.minor.size"] = 4
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["xtick.major.size"] = 8
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.minor.size"] = 4
    plt.rcParams["ytick.minor.width"] = 1
    plt.rcParams["ytick.major.size"] = 8
    plt.rcParams["ytick.major.width"] = 1

    red = "#EF3A47"
    blue = "#007BC0"
    black = "#000000"

    _, ax = plt.subplots(ncols=2, sharey=True, figsize=(22, 12))
    ax[0].hist(m_data, bins=bins, range=(1, 2), color=black, histtype="step", label="Data")
    ax[1].hist(m_data, bins=bins, range=(1, 2), color=black, histtype="step", label="Data")
    ax[0].hist(
        m_accmc,
        weights=tot_weights,
        bins=bins,
        range=(1, 2),
        color=black,
        alpha=0.1,
        label="Fit (unbinned)",
    )
    ax[1].hist(
        m_accmc,
        weights=tot_weights,
        bins=bins,
        range=(1, 2),
        color=black,
        alpha=0.1,
        label="Fit (unbinned)",
    )
    ax[0].hist(
        m_accmc,
        weights=s0p_weights,
        bins=bins,
        range=(1, 2),
        color=blue,
        alpha=0.1,
        label="$S_0^+$ (unbinned)",
    )
    ax[1].hist(
        m_accmc,
        weights=d2p_weights,
        bins=bins,
        range=(1, 2),
        color=red,
        alpha=0.1,
        label="$D_2^+$ (unbinned)",
    )
    centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    ax[0].scatter(centers, binned_tot_res, color=black, label="Fit (binned)")
    ax[1].scatter(centers, binned_tot_res, color=black, label="Fit (binned)")
    ax[0].scatter(centers, binned_s0p_res, color=blue, label="$S_0^+$ (binned)")
    ax[1].scatter(centers, binned_d2p_res, color=red, label="$D_2^+$ (binned)")

    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylim(0)
    ax[1].set_ylim(0)
    ax[0].set_xlabel("Mass of $K_S^0 K_S^0$ (GeV/$c^2$)")
    ax[1].set_xlabel("Mass of $K_S^0 K_S^0$ (GeV/$c^2$)")
    bin_width = int(1000 / bins)
    ax[0].set_ylabel(f"Counts / {bin_width} MeV/$c^2$")
    ax[1].set_ylabel(f"Counts / {bin_width} MeV/$c^2$")
    plt.tight_layout()
    plt.savefig("example_1.svg")


def fit_binned(bins: int, data_file: str, mc_file: str):
    res_mass = ld.Mass([2, 3])
    angles = ld.Angles(0, [1], [2], [2, 3])
    polarization = ld.Polarization(0, [1])
    data_ds_binned = ld.open_binned(data_file, res_mass, bins, (1.0, 2.0))
    accmc_ds_binned = ld.open_binned(mc_file, res_mass, bins, (1.0, 2.0))
    manager = ld.Manager()
    z00p = manager.register(ld.Zlm("Z00+", 0, 0, "+", angles, polarization))
    z22p = manager.register(ld.Zlm("Z22+", 2, 2, "+", angles, polarization))
    s0p = manager.register(ld.Scalar("S0+", ld.parameter("S0+ re")))
    d2p = manager.register(ld.ComplexScalar("D2+", ld.parameter("D2+ re"), ld.parameter("D2+ im")))
    pos_re = (s0p * z00p.real() + d2p * z22p.real()).norm_sqr()
    pos_im = (s0p * z00p.imag() + d2p * z22p.imag()).norm_sqr()
    model = pos_re + pos_im

    tot_res = []
    s0p_res = []
    d2p_res = []

    rng = np.random.default_rng()

    for ibin in range(bins):
        nll = ld.NLL(manager, data_ds_binned[ibin], accmc_ds_binned[ibin], model)
        p0 = rng.uniform(-1000.0, 1000.0, len(nll.parameters))
        status = nll.minimize(p0)

        tot_res.append(nll.project(status.x).sum())
        nll.isolate(["Z00+", "S0+"])
        s0p_res.append(nll.project(status.x).sum())
        nll.isolate(["Z22+", "D2+"])
        d2p_res.append(nll.project(status.x).sum())
    return (tot_res, s0p_res, d2p_res, data_ds_binned.edges)


def fit_unbinned(data_file: str, mc_file: str):
    res_mass = ld.Mass([2, 3])
    angles = ld.Angles(0, [1], [2], [2, 3])
    polarization = ld.Polarization(0, [1])
    data_ds = ld.open(data_file)
    accmc_ds = ld.open(mc_file)
    manager = ld.Manager()
    z00p = manager.register(ld.Zlm("Z00+", 0, 0, "+", angles, polarization))
    z22p = manager.register(ld.Zlm("Z22+", 2, 2, "+", angles, polarization))
    bw_f01500 = manager.register(
        ld.BreitWigner(
            "f0(1500)", ld.constant(1.506), ld.parameter("f0_width"), 0, ld.Mass([2]), ld.Mass([3]), res_mass
        )
    )
    bw_f21525 = manager.register(
        ld.BreitWigner(
            "f2(1525)", ld.constant(1.517), ld.parameter("f2_width"), 2, ld.Mass([2]), ld.Mass([3]), res_mass
        )
    )
    s0p = manager.register(ld.Scalar("S0+", ld.parameter("S0+ re")))
    d2p = manager.register(ld.ComplexScalar("D2+", ld.parameter("D2+ re"), ld.parameter("D2+ im")))
    pos_re = (s0p * bw_f01500 * z00p.real() + d2p * bw_f21525 * z22p.real()).norm_sqr()
    pos_im = (s0p * bw_f01500 * z00p.imag() + d2p * bw_f21525 * z22p.imag()).norm_sqr()
    model = pos_re + pos_im

    nll = ld.NLL(manager, data_ds, accmc_ds, model)
    p0 = [0.8, 0.5, 100, 50, 50]
    bounds = [
        (0.001, 1.0),
        (0.001, 1.0),
        (-1000.0, 1000.0),
        (-1000.0, 1000.0),
        (-1000.0, 1000.0),
    ]
    status = nll.minimize(p0, bounds=bounds)

    tot_weights = nll.project(status.x)
    nll.isolate(["S0+", "Z00+", "f0(1500)"])
    s0p_weights = nll.project(status.x)
    nll.isolate(["D2+", "Z22+", "f2(1525)"])
    d2p_weights = nll.project(status.x)

    return (tot_weights, s0p_weights, d2p_weights, status, nll.parameters)


if __name__ == "__main__":
    args = docopt(__doc__)  # pyright: ignore
    main(int(args["-n"]))
