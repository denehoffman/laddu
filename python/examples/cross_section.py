"""
Fit four accepted K_S K_S datasets and combine their differential cross sections.

After entering the direnv-managed development shell and running `just
python-dev` once, run the practical default example with:

    just cross-section-example cpu

Use `just cross-section-example-quick cpu` for an API smoke test or
`just cross-section-example-full jit` for a higher-statistics study.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import laddu as ld
import matplotlib as mpl
import numpy as np
from closure import (
    F2_MAGNITUDE_TRUTH,
    F2_PHASE_TRUTH,
    build_channel,
    build_model,
    fit_likelihood,
    print_generation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

mpl.use('Agg')
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Period:
    """Fabricated run-period inputs for the simultaneous analysis."""

    name: str
    label: str
    luminosity: float
    data_events: int
    acceptance_at_1_gev: float
    acceptance_at_2_gev: float


PERIODS = (
    Period('period_a', 'Period A', 18.0, 400, 0.040, 0.070),
    Period('period_b', 'Period B', 25.0, 560, 0.045, 0.067),
    Period('period_c', 'Period C', 34.0, 760, 0.036, 0.074),
    Period('period_d', 'Period D', 29.0, 660, 0.042, 0.071),
)
COMPONENTS: dict[str, Sequence[str]] = {'f0': ['f0'], 'f2': ['f2']}
COLORS = {'total': '#d62728', 'f0': '#1f77b4', 'f2': '#2ca02c'}


def acceptance_expr(mass: ld.Expr, period: Period) -> ld.Expr:
    """Linear toy efficiency across the generated mass range."""
    slope = period.acceptance_at_2_gev - period.acceptance_at_1_gev
    return period.acceptance_at_1_gev + slope * (mass - 1.0)


def acceptance_values(masses: np.ndarray, period: Period) -> np.ndarray:
    """Numerical form of the toy efficiency used for MC rejection."""
    slope = period.acceptance_at_2_gev - period.acceptance_at_1_gev
    return np.clip(period.acceptance_at_1_gev + slope * (masses - 1.0), 0.0, 1.0)


def subset_dataset(
    dataset: ld.Dataset,
    mask: np.ndarray,
    execution: ld.Execution,
) -> ld.Dataset:
    """Copy selected resident events while preserving columns and weights."""
    p4s: dict[str, np.ndarray] = {}
    for name in dataset.p4_names():
        p4 = ld.Vec4.event(name)
        p4s[name] = np.column_stack(
            [
                dataset.evaluate(component, execution=execution, real=True)
                for component in (p4.e(), p4.px(), p4.py(), p4.pz())
            ]
        )[mask]
    scalars = {
        name: np.asarray(
            dataset.evaluate(ld.scalar(name), execution=execution, real=True),
            dtype=float,
        )[mask]
        for name in dataset.scalar_names()
    }
    weights = np.asarray(dataset.weights(), dtype=float)[mask]
    return ld.Dataset.from_arrays(p4s=p4s, scalars=scalars, weights=weights)


def accept_mc(
    generated: ld.Dataset,
    mass: ld.Expr,
    period: Period,
    execution: ld.Execution,
    seed: int,
) -> ld.Dataset:
    """Apply a mass-dependent Bernoulli acceptance to phase-space MC."""
    masses = np.asarray(generated.evaluate(mass, execution=execution, real=True), dtype=float)
    rng = np.random.default_rng(seed)
    mask = rng.random(len(masses)) < acceptance_values(masses, period)
    if not np.any(mask):
        msg = f'{period.label} accepted MC is empty; increase --generated-mc-events'
        raise RuntimeError(msg)
    return subset_dataset(generated, mask, execution)


def binned_std(estimate: ld.BinnedEstimate) -> np.ndarray:
    """Return the sample standard deviation of a propagated binned estimate."""
    return np.std(np.asarray(estimate.draws, dtype=float), axis=0, ddof=1)


def scalar_text(estimate: ld.Estimate) -> str:
    """Format a central value and bootstrap standard deviation."""
    return f'{estimate.central:.5g} +/- {estimate.std():.2g} pb'


def integrate_binned(estimate: ld.BinnedEstimate, widths: np.ndarray) -> ld.Estimate:
    """Integrate a one-dimensional differential estimate over its bins."""
    central = float(np.dot(np.asarray(estimate.central, dtype=float), widths))
    draws = np.asarray(estimate.draws, dtype=float) @ widths
    return ld.Estimate(central, draws=draws.tolist())


def plot_acceptance_diagnostics(
    periods: tuple[Period, ...],
    generated_samples: list[ld.Dataset],
    accepted_samples: list[ld.Dataset],
    distributions: list[ld.DifferentialCrossSection],
    mass: ld.Expr,
    execution: ld.Execution,
    edges: np.ndarray,
    output: Path,
) -> None:
    """Plot corrected yields and empirical/nominal acceptance for every period."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True, constrained_layout=True)

    for axis, period, generated, accepted, distribution in zip(
        axes.flat,
        periods,
        generated_samples,
        accepted_samples,
        distributions,
        strict=True,
    ):
        generated_mass = np.asarray(
            generated.evaluate(mass, execution=execution, real=True),
            dtype=float,
        )
        accepted_mass = np.asarray(
            accepted.evaluate(mass, execution=execution, real=True),
            dtype=float,
        )
        generated_counts, _ = np.histogram(
            generated_mass,
            bins=edges,
            weights=np.asarray(generated.weights(), dtype=float),
        )
        accepted_counts, _ = np.histogram(
            accepted_mass,
            bins=edges,
            weights=np.asarray(accepted.weights(), dtype=float),
        )
        empirical_acceptance = np.divide(
            accepted_counts,
            generated_counts,
            out=np.zeros_like(accepted_counts),
            where=generated_counts > 0.0,
        )

        corrected_yield = np.asarray(distribution.data.central, dtype=float) * period.luminosity
        corrected_error = binned_std(distribution.data) * period.luminosity
        axis.errorbar(
            centers,
            corrected_yield,
            xerr=0.5 * widths,
            yerr=corrected_error,
            fmt='o',
            color='black',
            markersize=3.5,
            linewidth=1.0,
            label='Corrected data yield',
        )
        axis.set(
            title=rf'$\mathrm{{{period.label}}}$',
            ylabel=r'Acceptance-corrected events / GeV',
            ylim=(0.0, None),
        )
        axis.tick_params(direction='in', top=True)

        acceptance_axis = axis.twinx()
        acceptance_axis.stairs(
            empirical_acceptance,
            edges,
            color='#9467bd',
            linewidth=1.4,
            label='Accepted / generated MC',
        )
        acceptance_axis.plot(
            centers,
            acceptance_values(centers, period),
            color='#ff7f0e',
            linestyle='--',
            linewidth=1.4,
            label='Injected acceptance',
        )
        acceptance_axis.set(
            ylabel='Acceptance',
            ylim=(0.0, 0.10),
        )
        acceptance_axis.tick_params(direction='in', right=True)

        handles, labels = axis.get_legend_handles_labels()
        right_handles, right_labels = acceptance_axis.get_legend_handles_labels()
        axis.legend(handles + right_handles, labels + right_labels, frameon=False, fontsize=8)

    for axis in axes[-1]:
        axis.set_xlabel(r'$m(K_S^0 K_S^0)\ [\mathrm{GeV}]$')
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_combined_cross_section(
    distribution: ld.DifferentialCrossSection,
    output: Path,
) -> None:
    """Plot combined data and fitted differential cross sections with 1-sigma errors."""
    edges = np.asarray(distribution.edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    data = np.asarray(distribution.data.central, dtype=float)
    data_error = binned_std(distribution.data)

    figure, axis = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    axis.errorbar(
        centers,
        data,
        xerr=0.5 * widths,
        yerr=data_error,
        fmt='o',
        color='black',
        markersize=4,
        linewidth=1,
        capsize=0,
        label=r'$\mathrm{Acceptance\ corrected\ data}$',
        zorder=5,
    )

    estimates = [('total', distribution.model), *sorted(distribution.components.items())]
    labels = {
        'total': r'$\mathrm{Coherent\ total}$',
        'f0': r'$f_0(1500)$',
        'f2': r'$f_2(1270)$',
    }
    linestyles = {'total': '-', 'f0': '--', 'f2': ':'}
    linewidths = {'total': 2.2, 'f0': 1.8, 'f2': 1.8}
    for name, estimate in estimates:
        central = np.asarray(estimate.central, dtype=float)
        error = binned_std(estimate)
        color = COLORS[name]
        axis.fill_between(
            edges,
            np.append(central - error, central[-1] - error[-1]),
            np.append(central + error, central[-1] + error[-1]),
            step='post',
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        axis.stairs(
            central,
            edges,
            label=labels[name],
            color=color,
            linewidth=linewidths[name],
            linestyle=linestyles[name],
        )

    axis.set(
        xlabel=r'$m(K_S^0 K_S^0)\ [\mathrm{GeV}]$',
        ylabel=r'$d\sigma/dm\ [\mathrm{pb}/\mathrm{GeV}]$',
        title=r'$\gamma p \to K_S^0 K_S^0 p\;\mathrm{combined\ cross\ section}$',
        xlim=(edges[0], edges[-1]),
        ylim=(0.0, None),
    )
    axis.legend(frameon=False)
    axis.tick_params(direction='in', top=True, right=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-mc-events', type=int, default=30_000)
    parser.add_argument('--data-scale', type=float, default=1.0)
    parser.add_argument('--pilot-proposals', type=int, default=20_000)
    parser.add_argument('--max-proposals', type=int, default=100_000_000)
    parser.add_argument('--max-fit-steps', type=int, default=120)
    parser.add_argument('--bootstrap-samples', type=int, default=20)
    parser.add_argument('--projection-bins', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0x4352_4F53_5353_4543)
    parser.add_argument('--backend', choices=['cpu', 'jit', 'gpu'], default='cpu')
    parser.add_argument('--precision', choices=['auto', 'f32', 'f64'], default='auto')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--output', type=Path, default=Path('target/python-cross-section'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true', help='run a small API smoke test')
    mode.add_argument('--full', action='store_true', help='run a higher-statistics analysis')
    args = parser.parse_args()
    if args.quick:
        args.generated_mc_events = 1_500
        args.data_scale = 0.06
        args.max_fit_steps = 8
        args.bootstrap_samples = 4
        args.projection_bins = 16
    elif args.full:
        args.generated_mc_events = 200_000
        args.data_scale = 10.0
        args.max_fit_steps = 250
        args.bootstrap_samples = 100
        args.projection_bins = 50
    for name in (
        'generated_mc_events',
        'pilot_proposals',
        'max_proposals',
        'max_fit_steps',
        'bootstrap_samples',
        'projection_bins',
    ):
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.data_scale <= 0.0:
        parser.error('--data-scale must be positive')
    if args.bootstrap_samples < 2:  # noqa: PLR2004
        parser.error('--bootstrap-samples must be at least 2')
    return args


def main() -> None:  # noqa: PLR0915
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    periods = tuple(
        Period(
            period.name,
            period.label,
            period.luminosity,
            max(1, round(period.data_events * args.data_scale)),
            period.acceptance_at_1_gev,
            period.acceptance_at_2_gev,
        )
        for period in PERIODS
    )
    execution = ld.Execution(
        args.backend,
        precision=args.precision,
        autodiff='forward',
        threads=args.threads,
    )
    print(f'execution: {execution}', flush=True)
    print(
        f'analysis: {sum(period.data_events for period in periods):,} accepted data events, '
        f'{args.generated_mc_events:,} generated MC events per period, '
        f'{args.bootstrap_samples} joint bootstrap refits',
        flush=True,
    )

    channel = build_channel()
    mass = channel.mass('X')
    model = build_model(channel)
    generator = ld.Generator(channel)
    truth = {'f2_magnitude': F2_MAGNITUDE_TRUTH, 'f2_phase': F2_PHASE_TRUTH}
    generated_samples: list[ld.Dataset] = []
    accepted_samples: list[ld.Dataset] = []
    data_samples: list[ld.Dataset] = []
    terms: list[ld.NLL] = []

    generation_started = time.perf_counter()
    for index, period in enumerate(periods):
        period_seed = args.seed + 10_000 * index
        print(
            f'{period.label}: generating {args.generated_mc_events:,} phase-space MC events...',
            flush=True,
        )
        generated, generated_report = generator.weighted(
            args.generated_mc_events,
            execution=execution,
            memory='256 MiB',
            seed=period_seed,
        )
        accepted = accept_mc(
            generated,
            mass,
            period,
            execution,
            period_seed + 1,
        )
        print(
            f'{period.label}: accepted {len(accepted):,}/{len(generated):,} MC events '
            f'({100.0 * len(accepted) / len(generated):.2f}% raw)',
            flush=True,
        )

        print(
            f'{period.label}: generating {period.data_events:,} acceptance-folded model events...',
            flush=True,
        )
        accepted_model = build_model(channel, acceptance_expr(mass, period))
        data, data_report = generator.unweighted(
            period.data_events,
            accepted_model,
            parameters=truth,
            execution=execution,
            memory='256 MiB',
            seed=period_seed + 2,
            max_proposals=args.max_proposals,
            pilot_proposals=args.pilot_proposals,
            safety_factor=2.0,
            grow_envelope=True,
        )
        print_generation(f'{period.label} generated MC', generated_report)
        print_generation(f'{period.label} modeled data', data_report)
        generated_samples.append(generated)
        accepted_samples.append(accepted)
        data_samples.append(data)
        terms.append(ld.NLL(model, data=data, accepted_mc=accepted, name=period.name))

    generation_time = time.perf_counter() - generation_started
    print('preparing the four-term joint likelihood...', flush=True)
    likelihood = ld.Likelihood(terms, execution=execution)
    initial = likelihood.sample_parameters(seed=args.seed + 1)
    initial_nll = likelihood.value(initial)

    fit_started = time.perf_counter()
    print(f'fitting jointly for at most {args.max_fit_steps} steps...', flush=True)
    fit = fit_likelihood(likelihood, initial, args.max_fit_steps, progress=True)
    fit_time = time.perf_counter() - fit_started
    names = fit.parameter_names or likelihood.parameter_names
    fitted = dict(zip(names, np.asarray(fit.x, dtype=float), strict=True))
    print(f'joint fit completed in {fit_time:.3f}s', flush=True)

    bootstrap_started = time.perf_counter()
    print(f'running {args.bootstrap_samples} paired joint bootstrap refits...', flush=True)
    ensemble = likelihood.bootstrap_fit(
        args.bootstrap_samples,
        initial=fit.x,
        seed=args.seed + 2,
        terminators=[ld.ganesh.MaxSteps(args.max_fit_steps)],
    )
    bootstrap_time = time.perf_counter() - bootstrap_started
    print(f'bootstrap ensemble completed in {bootstrap_time:.3f}s', flush=True)

    cross_section_started = time.perf_counter()
    cross_sections = []
    for period, generated in zip(periods, generated_samples, strict=True):
        print(f'{period.label}: preparing cross-section integrals...', flush=True)
        cross_sections.append(
            likelihood.cross_section(
                period.name,
                generated_mc=generated,
                luminosity=period.luminosity,
                parameters=fitted,
                ensemble=ensemble,
            )
        )
    print('combining the four effective exposures...', flush=True)
    combined = ld.CrossSection.combine(cross_sections)
    cross_section_time = time.perf_counter() - cross_section_started
    print(f'cross-section preparation completed in {cross_section_time:.3f}s', flush=True)
    limits = (2.0 * channel.particle('ks1').mass, 2.0)
    edges = np.linspace(*limits, args.projection_bins + 1)
    axis = ld.Axis(mass, edges=edges)
    diagnostic_edges = np.linspace(*limits, min(16, args.projection_bins) + 1)
    diagnostic_axis = ld.Axis(mass, edges=diagnostic_edges)
    period_differential_started = time.perf_counter()
    distributions = []
    for period, cross_section in zip(periods, cross_sections, strict=True):
        print(f'{period.label}: propagating the projection set...', flush=True)
        projection_set = cross_section.projection_set(
            {'mass': diagnostic_axis},
            components=COMPONENTS,
        )
        distributions.append(projection_set['mass'])
    period_differential_time = time.perf_counter() - period_differential_started
    print(
        f'period differential cross sections completed in {period_differential_time:.3f}s',
        flush=True,
    )
    combined_differential_started = time.perf_counter()
    print('propagating the combined differential cross section...', flush=True)
    combined_distribution = combined.differential(axis, components=COMPONENTS)
    combined_differential_time = time.perf_counter() - combined_differential_started
    print(
        f'combined differential cross section completed in {combined_differential_time:.3f}s',
        flush=True,
    )

    diagnostic_widths = np.diff(diagnostic_edges)
    integrated = {
        period.name: {
            'total': integrate_binned(distribution.model, diagnostic_widths),
            'f0': integrate_binned(distribution.components['f0'], diagnostic_widths),
            'f2': integrate_binned(distribution.components['f2'], diagnostic_widths),
        }
        for period, distribution in zip(periods, distributions, strict=True)
    }
    widths = np.diff(edges)
    integrated['combined'] = {
        'total': integrate_binned(combined_distribution.model, widths),
        'f0': integrate_binned(combined_distribution.components['f0'], widths),
        'f2': integrate_binned(combined_distribution.components['f2'], widths),
    }

    print('\nintegrated cross sections', flush=True)
    print('sample       coherent total              f0(1500)              f2(1270)', flush=True)
    for period in periods:
        estimates = integrated[period.name]
        print(
            f'{period.label:<10} {scalar_text(estimates["total"]):>22} '
            f'{scalar_text(estimates["f0"]):>22} '
            f'{scalar_text(estimates["f2"]):>22}',
            flush=True,
        )
    combined_estimates = integrated['combined']
    print(
        f'Combined   {scalar_text(combined_estimates["total"]):>22} '
        f'{scalar_text(combined_estimates["f0"]):>22} '
        f'{scalar_text(combined_estimates["f2"]):>22}',
        flush=True,
    )
    print(
        '\nThe coherent total includes interference and need not equal the sum of components.',
        flush=True,
    )

    diagnostic_path = args.output / 'acceptance-diagnostics.png'
    combined_path = args.output / 'combined-cross-section.png'
    plotting_started = time.perf_counter()
    print('making acceptance and cross-section plots...', flush=True)
    plot_acceptance_diagnostics(
        periods,
        generated_samples,
        accepted_samples,
        distributions,
        mass,
        execution,
        diagnostic_edges,
        diagnostic_path,
    )
    plot_combined_cross_section(combined_distribution, combined_path)
    plotting_time = time.perf_counter() - plotting_started
    print(f'plotting completed in {plotting_time:.3f}s', flush=True)

    print('writing generated MC, accepted MC, modeled data, and fit metadata...', flush=True)
    for period, generated, accepted, data in zip(
        periods,
        generated_samples,
        accepted_samples,
        data_samples,
        strict=True,
    ):
        generated.write_to(ld.ParquetSink(args.output / f'{period.name}-generated.parquet'))
        accepted.write_to(ld.ParquetSink(args.output / f'{period.name}-accepted.parquet'))
        data.write_to(ld.ParquetSink(args.output / f'{period.name}-data.parquet'))

    totals = {
        period.name: {
            'total_pb': integrated[period.name]['total'].central,
            'f0_pb': integrated[period.name]['f0'].central,
            'f2_pb': integrated[period.name]['f2'].central,
        }
        for period in periods
    }
    totals['combined'] = {
        'total_pb': combined_estimates['total'].central,
        'f0_pb': combined_estimates['f0'].central,
        'f2_pb': combined_estimates['f2'].central,
    }
    (args.output / 'fit.json').write_text(
        json.dumps(
            {
                'backend': args.backend,
                'truth': truth,
                'fitted': fitted,
                'initial_nll': initial_nll,
                'final_nll': fit.fx,
                'bootstrap_samples': args.bootstrap_samples,
                'periods': [
                    {
                        'name': period.name,
                        'luminosity_pb_inverse': period.luminosity,
                        'data_events': period.data_events,
                        'acceptance_at_1_gev': period.acceptance_at_1_gev,
                        'acceptance_at_2_gev': period.acceptance_at_2_gev,
                    }
                    for period in periods
                ],
                'integrated_cross_sections': totals,
                'timings_seconds': {
                    'generation': generation_time,
                    'fit': fit_time,
                    'bootstrap': bootstrap_time,
                    'cross_section_preparation': cross_section_time,
                    'period_differentials': period_differential_time,
                    'combined_differential': combined_differential_time,
                    'plotting': plotting_time,
                },
            },
            indent=2,
        )
        + '\n'
    )
    print(
        f'completed in {time.perf_counter() - generation_started:.3f}s; wrote outputs to {args.output}',
        flush=True,
    )


if __name__ == '__main__':
    main()
