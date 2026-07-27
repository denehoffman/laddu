"""
Generate and fit a small K_S K_S closure test with laddu's Python API.

After entering the direnv-managed development shell and running `just
python-dev` once, run the practical default example with:

    just example gpu

Use the full Rust-demo sample sizes with:

    just example-full jit
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import laddu as ld
import matplotlib as mpl
import numpy as np

mpl.use('Agg')
import matplotlib.pyplot as plt

F2_MAGNITUDE_TRUTH = 0.65
F2_PHASE_TRUTH = 0.7

# PDG 2025 values used by the Rust closure demo.
F0_MASS = 1.522
F0_WIDTH = 0.108
F2_MASS = 1.275_412_049_919_005
F2_WIDTH = 0.186_554_356_637_326_4


def build_channel() -> ld.Channel:
    """Build gamma p -> X p, X -> K_S K_S with generation metadata."""
    kaon = ld.particles.K_SHORT
    threshold = 2.0 * kaon.mass

    return ld.Channel(
        'gamma p -> K_S K_S p',
        edges=[
            ld.Edge(
                'gamma',
                p4='gamma',
                particle=ld.particles.PHOTON,
                output=True,
                initial_momentum=ld.InitialMomentum.uniform_energy(8.0, 9.0, [0.0, 0.0, 1.0]),
            ),
            ld.Edge(
                'target',
                p4='target',
                particle=ld.particles.PROTON,
                output=True,
                initial_momentum=ld.InitialMomentum.momentum([0.0, 0.0, 0.0]),
            ),
            ld.Edge('X', mass_proposal=ld.MassProposal(threshold, 2.0)),
            ld.Edge('recoil', p4='recoil', particle=ld.particles.PROTON, output=True),
            ld.Edge('ks1', p4='ks1', particle=kaon, output=True),
            ld.Edge('ks2', p4='ks2', particle=kaon, output=True),
        ],
        vertices=[
            ld.Vertex(
                'production',
                incoming=['gamma', 'target'],
                outgoing=['X', 'recoil'],
                generation=ld.VertexProposal.t_exchange('gamma', 'X', slope=4.0, uniform_fraction=0.2),
            ),
            ld.Vertex(
                'decay',
                incoming=['X'],
                outgoing=['ks1', 'ks2'],
                generation=ld.VertexProposal.isotropic(),
            ),
        ],
    )


def production_total_j(channel: ld.Channel, resonance: ld.Particle) -> ld.J:
    """Choose the common initial/final spin coupling used by the Rust demo."""
    initial = channel.particle('gamma').spin.coupled_with(
        channel.particle('target').spin,
    )
    final = resonance.spin.coupled_with(channel.particle('recoil').spin)
    try:
        return next(candidate for candidate in initial if candidate in final)
    except StopIteration as error:
        msg = 'production vertex has no allowed total-spin coupling'
        raise ValueError(msg) from error


def sequential_wave(
    channel: ld.Channel,
    resonance: ld.Particle,
    *,
    photon_helicity: ld.M,
    target_helicity: ld.M,
    recoil_helicity: ld.M,
    first_kaon_helicity: ld.M,
    second_kaon_helicity: ld.M,
    line_shape: ld.Expr,
) -> ld.Expr:
    """Construct the same sequential-helicity wave used by the Rust demo."""
    production = channel.vertex('production')
    decay = channel.vertex('decay')
    beam_axis = production.vec3('gamma')
    helicity_axis = production.vec3('X')
    production_normal = beam_axis.cross(helicity_axis)
    y_axis = ld.Vec3.y_axis()
    production_theta = production.theta('X', beam_axis, y_axis)
    production_phi = production.phi('X', beam_axis, y_axis)
    decay_theta = decay.theta('ks1', helicity_axis, production_normal)
    decay_phi = decay.phi('ks1', helicity_axis, production_normal)

    resonance_spin = resonance.spin
    photon_spin = channel.particle('gamma').spin
    target_spin = channel.particle('target').spin
    recoil_spin = channel.particle('recoil').spin
    first_kaon_spin = channel.particle('ks1').spin
    second_kaon_spin = channel.particle('ks2').spin
    production_spin = production_total_j(channel, resonance)
    if first_kaon_spin != ld.S(0) or second_kaon_spin != ld.S(0):
        msg = 'this K_S K_S wave requires two spin-zero daughters'
        raise ValueError(msg)
    decay_spin = ld.S(0)
    decay_orbital = ld.L(resonance_spin)
    decay_helicity = first_kaon_helicity - second_kaon_helicity
    initial_projection = photon_helicity - target_helicity
    initial_coupling = ld.clebsch_gordan(
        photon_spin,
        photon_helicity,
        target_spin,
        -target_helicity,
        production_spin,
        initial_projection,
    )

    angular = ld.Expr(0.0)
    for resonance_helicity in resonance_spin.projections():
        final_projection = resonance_helicity - recoil_helicity
        production_coupling = ld.clebsch_gordan(
            resonance_spin,
            resonance_helicity,
            recoil_spin,
            -recoil_helicity,
            production_spin,
            final_projection,
        )
        daughter_spin_coupling = ld.clebsch_gordan(
            first_kaon_spin,
            first_kaon_helicity,
            second_kaon_spin,
            -second_kaon_helicity,
            decay_spin,
            decay_helicity,
        )
        orbital_coupling = ld.clebsch_gordan(
            decay_orbital,
            ld.M(0),
            decay_spin,
            decay_helicity,
            resonance_spin,
            decay_helicity,
        )
        coefficient = initial_coupling * production_coupling * daughter_spin_coupling * orbital_coupling
        if coefficient == 0.0:
            continue

        production_d = (
            ld.WignerD(
                production_spin,
                initial_projection,
                final_projection,
            )
            .D(production_phi, production_theta)
            .conj()
        )
        decay_d = (
            ld.WignerD(
                resonance_spin,
                resonance_helicity,
                decay_helicity,
            )
            .D(decay_phi, decay_theta)
            .conj()
        )
        angular += coefficient * production_d * decay_d

    normalization = math.sqrt(production_spin.multiplicity * resonance_spin.multiplicity) / (4.0 * math.pi)
    ls_normalization = math.sqrt(decay_orbital.multiplicity / resonance_spin.multiplicity)
    return normalization * ls_normalization * line_shape * angular


def build_model(channel: ld.Channel, efficiency: ld.Expr | None = None) -> ld.Model:
    """Build the Rust demo's unpolarized sequential-helicity intensity."""
    s = channel.s('X')
    kaon_mass = channel.particle('ks1').mass

    f0_particle = ld.Particle('f0(1500)', spin=ld.S(0), parity=ld.Parity.POSITIVE, mass=F0_MASS)
    f2_particle = ld.Particle('f2(1270)', spin=ld.S(2), parity=ld.Parity.POSITIVE, mass=F2_MASS)
    f0 = ld.relativistic_breit_wigner(s, f0_particle.mass, F0_WIDTH, kaon_mass, kaon_mass, l=0)
    f2 = ld.relativistic_breit_wigner(s, f2_particle.mass, F2_WIDTH, kaon_mass, kaon_mass, l=2)

    magnitude = ld.parameter('f2_magnitude', initial=0.35, bounds=(0.0, 2.0), scale=0.5)
    phase = ld.parameter(
        'f2_phase',
        initial=0.0,
        bounds=(-math.pi, math.pi),
        periodic=True,
        scale=1.0,
    )
    f2_coupling = ld.polar_complex(magnitude, phase)

    photon_spin = channel.particle('gamma').spin
    target_spin = channel.particle('target').spin
    recoil_spin = channel.particle('recoil').spin
    first_kaon_spin = channel.particle('ks1').spin
    second_kaon_spin = channel.particle('ks2').spin
    photon_helicities = [helicity for helicity in photon_spin.projections() if helicity != ld.M(0)]

    coherent = ld.Expr(0.0)
    for photon_helicity in photon_helicities:
        for target_helicity in target_spin.projections():
            for recoil_helicity in recoil_spin.projections():
                for first_kaon_helicity in first_kaon_spin.projections():
                    for second_kaon_helicity in second_kaon_spin.projections():
                        f0_wave = sequential_wave(
                            channel,
                            f0_particle,
                            photon_helicity=photon_helicity,
                            target_helicity=target_helicity,
                            recoil_helicity=recoil_helicity,
                            first_kaon_helicity=first_kaon_helicity,
                            second_kaon_helicity=second_kaon_helicity,
                            line_shape=f0,
                        ).tagged('f0')
                        f2_wave = sequential_wave(
                            channel,
                            f2_particle,
                            photon_helicity=photon_helicity,
                            target_helicity=target_helicity,
                            recoil_helicity=recoil_helicity,
                            first_kaon_helicity=first_kaon_helicity,
                            second_kaon_helicity=second_kaon_helicity,
                            line_shape=f2,
                        ).tagged('f2')
                        coherent += (f0_wave + f2_coupling * f2_wave).norm_sqr()

    intensity = coherent * 0.25
    if efficiency is not None:
        intensity *= efficiency
    return ld.Model(intensity)


def wrapped_phase_residual(value: float, truth: float) -> float:
    return math.atan2(math.sin(value - truth), math.cos(value - truth))


def print_generation(label: str, report: ld.GenerationReport) -> None:
    print(
        f'{label}: {report.produced:,} events from {report.proposals:,} proposals '
        f'(acceptance {100.0 * report.acceptance_rate:.2f}%, '
        f'max weight {report.maximum_weight:.3e})',
        flush=True,
    )


def fit_likelihood(
    likelihood: ld.Likelihood,
    initial: list[float],
    max_steps: int,
    *,
    progress: bool = False,
) -> ld.ganesh.MinimizationSummary:
    """Fit one likelihood, optionally showing the central fit's progress."""
    observers = [ld.ganesh.ProgressObserver(interval=1)] if progress else []
    return likelihood.fit(
        ld.ganesh.LBFGSBConfig(history_size=10),
        initial=ld.ganesh.VectorInit(initial),
        terminators=[ld.ganesh.MaxSteps(max_steps)],
        observers=observers,
    )


def plot_closure(
    channel: ld.Channel,
    normalization: ld.Dataset,
    likelihood: ld.Likelihood,
    fitted: dict[str, float],
    bins: int,
    bootstrap_samples: int,
    bootstrap_fit_steps: int,
    seed: int,
    output: Path,
) -> dict[str, float]:
    """Plot bootstrap data errors and a refitted 68% projection band."""
    mass = channel.mass('X')
    limits = (2.0 * channel.particle('ks1').mass, 2.0)
    edges = np.linspace(*limits, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    print(f'running {bootstrap_samples} paired Poisson-bootstrap refits...', flush=True)
    ensemble = likelihood.bootstrap_fit(
        bootstrap_samples,
        ld.ganesh.LBFGSBConfig(history_size=10),
        initial=[fitted[name] for name in likelihood.parameter_names],
        seed=seed,
        terminators=[ld.ganesh.MaxSteps(bootstrap_fit_steps)],
    )
    print('bootstrap refits complete; propagating projection uncertainties...', flush=True)
    cross_section = likelihood.cross_section(
        'ksks',
        normalization,
        luminosity=1.0,
        parameters=fitted,
        ensemble=ensemble,
    )
    differential = cross_section.differential(
        ld.Axis(mass, edges),
        components={'f0': ['f0'], 'f2': ['f2']},
    )

    data_counts = np.asarray(differential.data.central, dtype=float) * widths
    data_draws = np.asarray(differential.data.draws, dtype=float) * widths
    fit_counts = np.asarray(differential.model.central, dtype=float) * widths
    fit_draws = np.asarray(differential.model.draws, dtype=float) * widths
    component_counts = {
        name: np.asarray(estimate.central, dtype=float) * widths for name, estimate in differential.components.items()
    }
    data_errors = np.std(data_draws, axis=0, ddof=1)
    fit_lower, fit_upper = np.quantile(fit_draws, [0.16, 0.84], axis=0)

    parameter_draws = np.asarray(ensemble.draws, dtype=float)
    parameter_columns = {name: parameter_draws[:, index] for index, name in enumerate(ensemble.parameter_names)}
    bootstrap_errors = {
        'f2_magnitude': float(np.std(parameter_columns['f2_magnitude'], ddof=1)),
        'f2_phase': float(
            np.std(
                [wrapped_phase_residual(value, fitted['f2_phase']) for value in parameter_columns['f2_phase']],
                ddof=1,
            )
        ),
    }

    figure, axis = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    axis.errorbar(
        centers,
        data_counts,
        xerr=0.5 * widths,
        yerr=data_errors,
        fmt='o',
        color='black',
        markersize=4,
        linewidth=1,
        capsize=0,
        label=r'$\mathrm{Pseudo\ data}$',
        zorder=4,
    )
    axis.fill_between(
        edges,
        np.append(fit_lower, fit_lower[-1]),
        np.append(fit_upper, fit_upper[-1]),
        step='post',
        color='#d62728',
        alpha=0.2,
        linewidth=0,
        label=r'$68\%\ \mathrm{bootstrap\ band}$',
    )
    projections = [
        (r'$\mathrm{Coherent\ fit}$', fit_counts, '#d62728', 2.2, '-'),
        (r'$f_0(1500)$', component_counts['f0'], '#1f77b4', 1.8, '--'),
        (r'$f_2(1270)$', component_counts['f2'], '#2ca02c', 1.8, ':'),
    ]
    for label, counts, color, linewidth, linestyle in projections:
        axis.stairs(
            counts,
            edges,
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )

    axis.set(
        xlabel=r'$m(K_S^0 K_S^0)\ [\mathrm{GeV}]$',
        ylabel=rf'$\mathrm{{Events}}\,/\,({widths[0]:.3f}\,\mathrm{{GeV}})$',
        title=r'$\gamma p \to K_S^0 K_S^0 p\;\mathrm{closure\ fit}$',
        xlim=limits,
        ylim=(0.0, None),
    )
    axis.legend(frameon=False)
    axis.tick_params(direction='in', top=True, right=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return bootstrap_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-events', type=int, default=1_000)
    parser.add_argument('--normalization-events', type=int, default=10_000)
    parser.add_argument('--pilot-proposals', type=int, default=20_000)
    parser.add_argument(
        '--max-proposals',
        type=int,
        default=100_000_000,
        help='fail instead of running indefinitely if unweighting is inefficient',
    )
    parser.add_argument('--max-fit-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0x0043_4C4F_5355_5245)
    parser.add_argument('--backend', choices=['cpu', 'jit', 'gpu'], default='cpu')
    parser.add_argument('--precision', choices=['auto', 'f32', 'f64'], default='auto')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--output', type=Path, default=Path('target/python-closure'))
    parser.add_argument('--projection-bins', type=int, default=50)
    parser.add_argument(
        '--bootstrap-samples',
        type=int,
        default=20,
        help='Poisson-bootstrap refits used for parameter errors and the plot band',
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--quick',
        action='store_true',
        help='use small samples and a short fit for a fast API smoke test',
    )
    mode.add_argument(
        '--full',
        action='store_true',
        help='use the Rust closure demo sample sizes and 100 bootstrap refits',
    )
    args = parser.parse_args()
    if args.quick:
        args.data_events = 25
        args.normalization_events = 250
        args.pilot_proposals = 20_000
        args.max_proposals = 10_000_000
        args.max_fit_steps = 5
        args.projection_bins = 20
        args.bootstrap_samples = 4
    elif args.full:
        args.data_events = 10_000
        args.normalization_events = 100_000
        args.pilot_proposals = 50_000
        args.max_fit_steps = 200
        args.projection_bins = 50
        args.bootstrap_samples = 100
    for name in (
        'data_events',
        'normalization_events',
        'pilot_proposals',
        'max_proposals',
        'max_fit_steps',
        'projection_bins',
        'bootstrap_samples',
    ):
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.bootstrap_samples < 2:  # noqa: PLR2004
        parser.error('--bootstrap-samples must be at least 2')
    return args


def main() -> None:  # noqa: PLR0915
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    execution = ld.Execution(
        args.backend,
        precision=args.precision,
        autodiff='forward',
        threads=args.threads,
    )
    print(f'execution: {execution}')
    print(
        f'bootstrap refits: {args.bootstrap_samples}',
        flush=True,
    )

    channel = build_channel()
    model = build_model(channel)
    generator = ld.Generator(channel)
    truth = {
        'f2_magnitude': F2_MAGNITUDE_TRUTH,
        'f2_phase': F2_PHASE_TRUTH,
    }

    started = time.perf_counter()
    print(f'generating {args.data_events:,} unweighted pseudo-events...', flush=True)
    data, data_report = generator.unweighted(
        args.data_events,
        model,
        parameters=truth,
        execution=execution,
        memory='256 MiB',
        seed=args.seed,
        max_proposals=args.max_proposals,
        pilot_proposals=args.pilot_proposals,
        safety_factor=2.0,
        grow_envelope=True,
    )
    data_time = time.perf_counter() - started

    started = time.perf_counter()
    print(
        f'generating {args.normalization_events:,} weighted normalization events...',
        flush=True,
    )
    normalization, normalization_report = generator.weighted(
        args.normalization_events,
        execution=execution,
        memory='256 MiB',
        seed=args.seed + 1,
    )
    normalization_time = time.perf_counter() - started

    started = time.perf_counter()
    print('preparing likelihood...', flush=True)
    likelihood = ld.Likelihood([ld.NLL(model, data, normalization, name='ksks')], execution=execution)
    initial = likelihood.sample_parameters(seed=args.seed + 2)
    initial_nll, initial_gradient = likelihood.value_and_gradient(initial)
    preparation_time = time.perf_counter() - started

    started = time.perf_counter()
    print(f'fitting central sample for at most {args.max_fit_steps} steps...', flush=True)
    fit = fit_likelihood(likelihood, initial, args.max_fit_steps, progress=True)
    fit_time = time.perf_counter() - started
    print(f'initial fit completed in {fit_time}s', flush=True)

    names = fit.parameter_names or likelihood.parameter_names
    fitted = dict(zip(names, np.asarray(fit.x, dtype=float), strict=True))
    magnitude = fitted['f2_magnitude']
    phase = fitted['f2_phase']

    started = time.perf_counter()
    plot_path = args.output / 'closure.png'
    bootstrap_errors = plot_closure(
        channel,
        normalization,
        likelihood,
        fitted,
        args.projection_bins,
        args.bootstrap_samples,
        args.max_fit_steps,
        args.seed + 10_000,
        plot_path,
    )
    projection_time = time.perf_counter() - started

    print_generation('pseudo-data', data_report)
    print_generation('normalization MC', normalization_report)
    print(
        'timings: '
        f'data {data_time:.3f}s, normalization MC {normalization_time:.3f}s, '
        f'likelihood preparation {preparation_time:.3f}s, fit {fit_time:.3f}s, '
        f'projection and plot {projection_time:.3f}s'
    )
    print(
        f'NLL: initial {initial_nll:.8e}, final {fit.fx:.8e}; '
        f'initial gradient norm {np.linalg.norm(initial_gradient):.8e}'
    )
    print('parameter       truth        fitted    bootstrap error      residual')
    print(
        f'f2_magnitude  {F2_MAGNITUDE_TRUTH:>10.6f}  {magnitude:>12.6f}'
        f'  {bootstrap_errors["f2_magnitude"]:>17.6f}'
        f'  {magnitude - F2_MAGNITUDE_TRUTH:>12.6f}'
    )
    print(
        f'f2_phase      {F2_PHASE_TRUTH:>10.6f}  {phase:>12.6f}'
        f'  {bootstrap_errors["f2_phase"]:>17.6f}'
        f'  {wrapped_phase_residual(phase, F2_PHASE_TRUTH):>12.6f}'
    )
    print(f'\n{fit}')

    data_path = args.output / 'pseudo-data.parquet'
    normalization_path = args.output / 'normalization.parquet'
    summary_path = args.output / 'fit.json'
    data.write_to(ld.ParquetSink(data_path))
    normalization.write_to(ld.ParquetSink(normalization_path))
    summary_path.write_text(
        json.dumps(
            {
                'backend': args.backend,
                'initial_nll': initial_nll,
                'final_nll': fit.fx,
                'truth': truth,
                'fitted': fitted,
                'bootstrap_samples': args.bootstrap_samples,
                'bootstrap_errors': bootstrap_errors,
                'timings_seconds': {
                    'data': data_time,
                    'normalization': normalization_time,
                    'likelihood_preparation': preparation_time,
                    'fit': fit_time,
                    'projection_and_plot': projection_time,
                },
            },
            indent=2,
        )
        + '\n'
    )
    print(f'wrote closure outputs and plot to {args.output}')


if __name__ == '__main__':
    main()
