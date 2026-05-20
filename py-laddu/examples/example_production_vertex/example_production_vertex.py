#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ganesh-rs",
#     "laddu",
#     "matplotlib",
#     "numpy",
#     "tqdm",
# ]
# ///
"""Compare mass-binned Zlm and explicit production-vertex fits."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import ganesh
import laddu as ld
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

P4_COLUMNS = ['beam', 'proton', 'kshort1', 'kshort2']
AUX_COLUMNS = ['pol_magnitude', 'pol_angle']
FRAME = 'GottfriedJackson'
PHOTON_HELICITIES = (-1, 1)
K_SECTORS = (0, 1)
PRODUCTION_SPIN_SECTORS = (
    (Fraction(1, 2), Fraction(1, 2)),
    (Fraction(-1, 2), Fraction(-1, 2)),
)
PHOTON = ld.ParticleProperties('gamma', spin=1)
PROTON = ld.ParticleProperties('proton', spin=Fraction(1, 2))
KSHORT = ld.ParticleProperties('K_S', spin=0)


def RESONANCE(spin: int) -> ld.ParticleProperties:
    return ld.ParticleProperties(f'X{spin}', spin=spin)


REACTION = ld.Reaction.two_to_two(
    ld.Particle.stored('beam'),
    ld.Particle.missing('target'),
    ld.Particle.composite(
        'kk', (ld.Particle.stored('kshort1'), ld.Particle.stored('kshort2'))
    ),
    ld.Particle.stored('proton'),
)
PRODUCTION = REACTION.production()
DECAY = REACTION.decay('kk')
POLARIZATION = REACTION.polarization(pol_magnitude='pol_magnitude', pol_angle='pol_angle')
MASS = REACTION.mass('kk')


@dataclass(frozen=True)
class DecayWave:
    label: str
    j: int
    m: int
    l: int


def decay_wave(label: str, *, j: int, m: int) -> DecayWave:
    allowed_waves = ld.allowed_partial_waves(
        RESONANCE(j),
        KSHORT,
        KSHORT,
        max_l=j,
        rules=ld.RuleSet.angular(),
    )
    return DecayWave(label, j=j, m=m, l=allowed_waves[0].wave.l)


WAVES: list[DecayWave] = [decay_wave('S0', j=0, m=0), decay_wave('D2', j=2, m=2)]


def complex_coupling(
    name: str,
    *tags: str,
    anchor: bool = False,
) -> ld.Expression:
    re = ld.parameter(f'{name}_re', initial=1.0)
    im = ld.parameter(f'{name}_im', initial=0.0)
    coupling = ld.ComplexScalar(*tags, re=re, im=im)
    if anchor:
        coupling.fix_parameter(f'{name}_im', 0.0)
    return coupling


def allowed_production_waves(
    decay_wave: DecayWave,
    max_l: int,
) -> list[ld.PartialWave]:
    waves = []
    for j_initial in ld.coupled_spins(PHOTON.spin, PROTON.spin):  # ty: ignore
        parent = ld.ParticleProperties(f'gamma_p_J{j_initial}', spin=j_initial)
        allowed_waves = ld.allowed_partial_waves(
            parent,
            RESONANCE(decay_wave.j),
            PROTON,
            max_l=max_l,
            rules=ld.RuleSet.angular(),
        )
        waves.extend([allowed.wave for allowed in allowed_waves])
    return waves


def decay_factor(decay_wave: DecayWave) -> ld.Expression:
    return DECAY.canonical_factor(
        spin=decay_wave.j,
        projection=decay_wave.m,
        orbital_l=decay_wave.l,
        coupled_spin=0,
        daughter='kshort1',
        daughter_1_spin=KSHORT.spin,  # ty: ignore
        daughter_2_spin=KSHORT.spin,  # ty: ignore
        lambda_1=0,
        lambda_2=0,
        frame=FRAME,
    )


def production_factor(
    photon_helicity: int,
    target_spin: Fraction,
    recoil_spin: Fraction,
    decay_wave: DecayWave,
    production_wave: ld.PartialWave,
) -> ld.Expression | None:
    projection = photon_helicity + target_spin
    lambda_total = decay_wave.m - recoil_spin
    if abs(projection) > production_wave.j or abs(lambda_total) > production_wave.j:
        return None
    return PRODUCTION.canonical_factor(
        spin=production_wave.j,
        projection=projection,
        orbital_l=production_wave.l,
        coupled_spin=production_wave.s,
        produced_spin=decay_wave.j,
        recoil_spin=PROTON.spin,  # ty: ignore
        lambda_produced=decay_wave.m,
        lambda_recoil=recoil_spin,
        frame=FRAME,
    )


def helicity_parameter_name(
    decay_wave: DecayWave,
    photon_helicity: int,
) -> str:
    return f'T_{decay_wave.label}_h{photon_helicity}'


def decay_waves_for_helicity(photon_helicity: int) -> list[DecayWave]:
    waves = []
    for wave in WAVES:
        m = wave.m if photon_helicity == 1 else -wave.m
        waves.append(DecayWave(wave.label, wave.j, m, wave.l))
    return waves


def production_amplitudes(
    target_spin: Fraction,
    recoil_spin: Fraction,
    max_l: int,
) -> dict[int, ld.Expression]:
    amplitudes = {}
    for helicity in PHOTON_HELICITIES:
        terms = []
        for decay_wave in decay_waves_for_helicity(helicity):
            decay = decay_factor(decay_wave)
            for production_wave in allowed_production_waves(decay_wave, max_l):
                prod = production_factor(
                    helicity,
                    target_spin,
                    recoil_spin,
                    decay_wave,
                    production_wave,
                )
                if prod is None:
                    continue
                coupling = complex_coupling(
                    helicity_parameter_name(decay_wave, helicity),
                    decay_wave.label,
                    anchor=decay_wave.label == 'S0',
                )
                terms.append(coupling * prod * decay)
        amplitudes[helicity] = ld.expr_sum(terms) if terms else ld.Zero()
    return amplitudes


def full_physics_model(max_l: int = 0) -> ld.Expression:
    terms = []
    for target_spin, recoil_spin in PRODUCTION_SPIN_SECTORS:
        amps = production_amplitudes(
            target_spin,
            recoil_spin,
            max_l,
        )
        for helicity in PHOTON_HELICITIES:
            for helicity_prime in PHOTON_HELICITIES:
                rho = ld.PhotonSDME(
                    helicity=helicity,
                    helicity_prime=helicity_prime,
                    polarization=POLARIZATION,
                )
                terms.append((rho * amps[helicity] * amps[helicity_prime].conj()).real())
    return ld.expr_sum(terms)


def helicity_t_parameter_name(
    wave: DecayWave,
    photon_helicity: int,
) -> str:
    return f'T_{wave.label}_h{photon_helicity}_m{wave.m}'


def helicity_t_model() -> ld.Expression:
    angles = DECAY.angles('kshort1', FRAME)
    terms = []
    for _ in K_SECTORS:
        amps = {}
        for helicity in PHOTON_HELICITIES:
            amp_terms = []
            for wave in WAVES:
                m = wave.m if helicity == 1 else -wave.m
                ylm = ld.Ylm(
                    l=wave.j,
                    m=m,
                    angles=angles,
                )
                t_wave = DecayWave(wave.label, wave.j, m, wave.l)
                coupling = complex_coupling(
                    helicity_t_parameter_name(t_wave, helicity),
                    wave.label,
                    anchor=wave.label == 'S0',
                )
                amp_terms.append(coupling * ylm)
            amps[helicity] = ld.expr_sum(amp_terms)

        for helicity in PHOTON_HELICITIES:
            for helicity_prime in PHOTON_HELICITIES:
                rho = ld.PhotonSDME(
                    helicity=helicity,
                    helicity_prime=helicity_prime,
                    polarization=POLARIZATION,
                )
                terms.append((rho * amps[helicity] * amps[helicity_prime].conj()).real())
    return ld.expr_sum(terms)


def zlm_parameter_name(wave: DecayWave, reflectivity: str) -> str:
    return f'Z_{wave.label}_{reflectivity}'


def zlm_model() -> ld.Expression:
    angles = DECAY.angles('kshort1', FRAME)
    sectors = []
    for _ in K_SECTORS:
        for reflectivity in ('+', '-'):
            real_terms = []
            imag_terms = []
            for wave in WAVES:
                zlm = ld.Zlm(
                    l=wave.j,
                    m=wave.m,
                    r=reflectivity,
                    angles=angles,
                    polarization=POLARIZATION,
                )
                coupling = complex_coupling(
                    zlm_parameter_name(wave, reflectivity),
                    wave.label,
                    f'{wave.label}{reflectivity}',
                    anchor=wave.label == 'S0',
                )
                real_terms.append(coupling * zlm.real())
                imag_terms.append(coupling * zlm.imag())
            sectors.append(
                ld.expr_sum(real_terms).norm_sqr() + ld.expr_sum(imag_terms).norm_sqr()
            )
    return ld.expr_sum(sectors)


def best_fit(
    model: ld.Expression, data: ld.Dataset, accmc: ld.Dataset, niters: int
) -> tuple[ld.NLL, ganesh.MinimizationSummary]:
    nll = ld.NLL(model, data, accmc)
    rng = np.random.default_rng(0)
    best = None
    best_fx = np.inf
    for _ in range(niters):
        p0 = rng.normal(0.0, 10.0, len(nll.parameters.free))
        fit = nll.minimize(p0)
        if fit.fx < best_fx:
            best = fit
            best_fx = fit.fx
    if best is None:
        msg = 'all fit attempts failed'
        raise RuntimeError(msg)
    return nll, best


def project_fit_counts(
    nll: ld.NLL,
    fit: ganesh.MinimizationSummary,
) -> dict[str, float]:
    weights = nll.project_weights(
        fit.x,
        subsets=[None, ['S0'], ['D2']],
        strict=True,
    )
    return {
        'total': float(weights[0].sum()),
        'S0': float(weights[1].sum()),
        'D2': float(weights[2].sum()),
    }


def print_fit_quality(rows: list[dict[str, float]]) -> None:
    print('\nFit quality')
    print(
        'bin  mass range       zlm NLL        helicity-T NLL  production NLL  '
        'Delta T    Delta prod'
    )
    for row in rows:
        print(
            f'{int(row["bin"]):>3}  '
            f'{row["mass_low"]:.3f}-{row["mass_high"]:.3f}  '
            f'{row["zlm_nll"]:>13.6g}  '
            f'{row["helicity_t_nll"]:>14.6g}  '
            f'{row["production_nll"]:>14.6g}  '
            f'{row["helicity_t_delta_nll"]:>8.3g}  '
            f'{row["production_delta_nll"]:>10.3g}'
        )


def print_projection_summary(rows: list[dict[str, float | str]]) -> None:
    print('\nProjected wave yields')
    print('bin  component  data       zlm        helicity-T  production')
    by_bin_component: dict[tuple[int, str], dict[str, float | str]] = {}
    for row in rows:
        key = (int(row['bin']), str(row['component']))
        entry = by_bin_component.setdefault(
            key,
            {
                'bin': int(row['bin']),
                'component': str(row['component']),
                'data_count': float(row['data_count']),
            },
        )
        entry[str(row['model'])] = float(row['projected_count'])
    component_order = {'total': 0, 'S0': 1, 'D2': 2}
    for key in sorted(
        by_bin_component, key=lambda item: (item[0], component_order[item[1]])
    ):
        row = by_bin_component[key]
        zlm = float(row.get('zlm', float('nan')))
        helicity_t = float(row.get('helicity_t', float('nan')))
        production = float(row.get('production', float('nan')))
        print(
            f'{int(row["bin"]):>3}  '
            f'{row["component"]!s:<9}  '
            f'{row["data_count"]:>7.0f}  '
            f'{zlm:>9.3g}  '
            f'{helicity_t:>10.3g}  '
            f'{production:>10.3g}'
        )


def plot_wave_projections(
    rows: list[dict[str, float | str]],
    output_path: Path,
    data_counts: np.ndarray,
    mass_edges: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = ['zlm', 'helicity_t', 'production']
    components = [
        ('total', 'total'),
        ('S0', 'S0'),
        ('D2', 'D2'),
    ]
    _, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4),
        sharex=True,
    )
    axes_flat = np.ravel(axes)
    offsets = {'zlm': -0.006, 'helicity_t': 0.0, 'production': 0.006}
    colors = {
        'zlm': '#000000',
        'helicity_t': '#0072B2',
        'production': '#D55E00',
    }
    for ax, (component, title) in zip(axes_flat, components, strict=False):
        ax.stairs(data_counts, mass_edges, color='#555555', label='data')
        for model in models:
            points = sorted(
                (
                    row
                    for row in rows
                    if row['component'] == component and row['model'] == model
                ),
                key=lambda row: float(row['mass_center']),
            )
            ax.scatter(
                [float(row['mass_center']) + offsets[model] for row in points],
                [float(row['projected_count']) for row in points],
                color=colors[model],
                label=model,
                s=28,
            )
        ax.set_title(title)
        ax.set_ylabel('projected counts')
        ax.grid(alpha=0.3)
    for ax in axes_flat:
        ax.set_xlabel(r'$m(K_S K_S)$ [GeV]')
    axes_flat[0].legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fit_bins(
    nbins: int,
    niters: int,
    *,
    plot_path: Path | None = None,
) -> None:
    script_dir = Path(os.path.realpath(__file__)).parent.resolve()
    data_dir = script_dir.parent / 'data'
    data = ld.io.read_parquet(data_dir / 'data.parquet', p4s=P4_COLUMNS, aux=AUX_COLUMNS)
    accmc = ld.io.read_parquet(
        data_dir / 'accmc.parquet', p4s=P4_COLUMNS, aux=AUX_COLUMNS
    )
    mass_range = (1.0, 2.0)
    data_masses = MASS.value_on(data)
    data_counts, mass_edges = np.histogram(data_masses, bins=nbins, range=mass_range)
    data_bins = data.bin_by(MASS, nbins, (1.0, 2.0))
    accmc_bins = accmc.bin_by(MASS, nbins, (1.0, 2.0))
    z_model = zlm_model()
    t_model = helicity_t_model()
    production_model = full_physics_model()
    fit_rows = []
    projection_rows = []
    for ibin in trange(nbins):
        mass_low = 1.0 + ibin / nbins
        mass_high = 1.0 + (ibin + 1) / nbins
        mass_center = 0.5 * (mass_low + mass_high)
        z_nll, z_fit = best_fit(z_model, data_bins[ibin], accmc_bins[ibin], niters)
        t_nll, t_fit = best_fit(t_model, data_bins[ibin], accmc_bins[ibin], niters)
        prod_nll, prod_fit = best_fit(
            production_model, data_bins[ibin], accmc_bins[ibin], niters
        )
        fit_rows.append(
            {
                'bin': ibin,
                'mass_low': mass_low,
                'mass_high': mass_high,
                'zlm_nll': z_fit.fx,
                'helicity_t_nll': t_fit.fx,
                'production_nll': prod_fit.fx,
                'helicity_t_delta_nll': t_fit.fx - z_fit.fx,
                'production_delta_nll': prod_fit.fx - z_fit.fx,
            }
        )
        projections_by_model = {
            'zlm': project_fit_counts(z_nll, z_fit),
            'helicity_t': project_fit_counts(t_nll, t_fit),
            'production': project_fit_counts(prod_nll, prod_fit),
        }
        for model_name, projections in projections_by_model.items():
            for component, projected_count in projections.items():
                reference = projections_by_model['zlm'][component]
                projection_rows.append(
                    {
                        'bin': ibin,
                        'mass_center': mass_center,
                        'model': model_name,
                        'component': component,
                        'projected_count': projected_count,
                        'delta_from_zlm': projected_count - reference,
                        'data_count': int(data_counts[ibin]),
                    }
                )
    print_fit_quality(fit_rows)
    print_projection_summary(projection_rows)
    if plot_path is not None:
        plot_wave_projections(projection_rows, plot_path, data_counts, mass_edges)
        print(f'\nWrote plot to {plot_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nbins', type=int, default=4)
    parser.add_argument('-i', '--niters', type=int, default=3)
    parser.add_argument(
        '--plot',
        type=Path,
        default=Path('production_vertex_projections.svg'),
        help='SVG path for the wave projection plot. Use "none" to disable.',
    )
    args = parser.parse_args()
    plot_path = None if str(args.plot).lower() == 'none' else args.plot
    fit_bins(args.nbins, args.niters, plot_path=plot_path)
