#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "laddu-mpi",
#     "numpy",
# ]
# ///
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import laddu as ld


@dataclass
class Summary:
    """Container with the metrics we want to report from the analysis."""

    total_events: int
    weighted_events: float
    root_local_events: int
    weighted_mass_mean: float
    weighted_mass_std: float
    min_mass: float
    max_mass: float


def main() -> None:
    use_mpi = (
        os.getenv('OMPI_WORLD_SIZE') is not None or os.getenv('PMI_SIZE') is not None
    )

    with ld.mpi.MPI(trigger=use_mpi and ld.mpi.is_mpi_available()):
        summary = run_analysis()
        report(summary)


def run_analysis() -> Summary:
    dataset = load_dataset()

    mass_variable = ld.Mass(['kshort1', 'kshort2'])
    masses = mass_variable.value_on(dataset)

    weighted_mean = float(np.average(masses, weights=dataset.weights))
    weighted_variance = float(
        np.average((masses - weighted_mean) ** 2, weights=dataset.weights)
    )

    return Summary(
        total_events=len(dataset),
        weighted_events=float(dataset.n_events_weighted),
        root_local_events=len(dataset.events),
        weighted_mass_mean=weighted_mean,
        weighted_mass_std=float(np.sqrt(max(weighted_variance, 0.0))),
        min_mass=float(masses.min()),
        max_mass=float(masses.max(initial=0.0)),
    )


def load_dataset() -> ld.Dataset:
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir / 'data.parquet'
    return ld.Dataset.open(
        data_file,
        p4s=['beam', 'proton', 'kshort1', 'kshort2'],
        boost_to_restframe_of=['proton', 'kshort1', 'kshort2'],
    )


def report(summary: Summary) -> None:
    if not ld.mpi.is_root():
        return

    size = ld.mpi.get_size() or 1

    print()
    print('MPI Demo Summary')
    print('----------------')
    if ld.mpi.using_mpi():
        rank = ld.mpi.get_rank() or 0
        print(f'MPI size        : {size}')
        print(f'Root rank       : {rank}')
    else:
        print('MPI size        : 1 (disabled)')
    print(f'Total events    : {summary.total_events}')
    print(f'Weighted events : {summary.weighted_events:.2f}')
    print(f'Root holds      : {summary.root_local_events} events')
    print(f'Mass mean       : {summary.weighted_mass_mean:.4f} GeV')
    print(f'Mass std dev    : {summary.weighted_mass_std:.4f} GeV')
    print(f'Mass range      : [{summary.min_mass:.4f}, {summary.max_mass:.4f}] GeV')

    if ld.mpi.using_mpi():
        print(
            '\nTip: re-run with a different number of ranks to watch the '
            'per-rank event counts change while the global metrics stay stable.'
        )
    else:
        print(
            "\nTip: launch the script with 'mpirun -n 4' to see laddu "
            'distribute the workload automatically.'
        )


if __name__ == '__main__':
    main()
