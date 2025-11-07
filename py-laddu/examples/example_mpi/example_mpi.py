#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "laddu-mpi",
#     "numpy",
# ]
# ///
r"""Analyse a stored Laddu dataset under MPI.

This example focuses on how to activate MPI support in ``laddu``. We load a
pre-generated sample of :math:`\gamma p \to K_{S}^{0} K_{S}^{0} p` events (the
same data used in ``example_2``), compute a weighted mass spectrum, and report
summary statistics. When executed under ``mpirun``, Laddu partitions the events
across ranks automatically, so the analysis code stays the same whether MPI is
active or not.

Run the script with ``mpirun`` to see MPI in action::

    mpirun -n 4 python example_mpi.py

Use ``--no-mpi`` to compare the serial behaviour::

    python example_mpi.py --no-mpi
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a sample Laddu dataset and show how to enable MPI with the provided "
            "context manager."
        )
    )
    parser.add_argument(
        "--no-mpi",
        action="store_true",
        help="Skip enabling MPI even if laddu has MPI support.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wants_mpi = not args.no_mpi
    mpi_available = False
    try:
        mpi_available = ld.mpi.is_mpi_available()
    except ModuleNotFoundError:  # pragma: no cover - only hit on non-MPI builds
        mpi_available = False

    if wants_mpi and not mpi_available:
        raise SystemExit(
            "laddu was built without MPI support. Install the laddu-mpi package to "
            "run this example with MPI."
        )

    context = ld.mpi.MPI(trigger=True) if wants_mpi and mpi_available else nullcontext()
    with context:
        summary = run_analysis()
        report(summary)


def run_analysis() -> Summary:
    dataset = load_dataset()

    weights = np.asarray(dataset.weights, dtype=float)
    mass_variable = ld.Mass(["kshort1", "kshort2"])
    masses = np.asarray(mass_variable.value_on(dataset), dtype=float)

    weighted_mean = float(np.average(masses, weights=weights))
    weighted_variance = float(np.average((masses - weighted_mean) ** 2, weights=weights))

    return Summary(
        total_events=len(dataset),
        weighted_events=float(dataset.n_events_weighted),
        root_local_events=len(dataset.events),
        weighted_mass_mean=weighted_mean,
        weighted_mass_std=float(np.sqrt(max(weighted_variance, 0.0))),
        min_mass=float(masses.min(initial=0.0)),
        max_mass=float(masses.max(initial=0.0)),
    )


def load_dataset() -> ld.Dataset:
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir / "data.parquet"
    p4_columns = ["beam", "proton", "kshort1", "kshort2"]
    aux_columns = ["pol_magnitude", "pol_angle"]
    boost_columns = p4_columns[1:]
    return ld.Dataset.open(
        data_file,
        p4s=p4_columns,
        aux=aux_columns,
        boost_to_restframe_of=boost_columns,
    )


def report(summary: Summary) -> None:
    using_mpi = _call_mpi(ld.mpi.using_mpi, default=False)
    is_root = True if not using_mpi else _call_mpi(ld.mpi.is_root, default=True)
    if not is_root:
        return

    size = _call_mpi(ld.mpi.get_size, default=1)

    print()
    print("MPI Demo Summary")
    print("----------------")
    if using_mpi:
        rank = _call_mpi(ld.mpi.get_rank, default=0)
        print(f"MPI size        : {size}")
        print(f"Root rank       : {rank}")
    else:
        print("MPI size        : 1 (disabled)")
    print(f"Total events    : {summary.total_events}")
    print(f"Weighted events : {summary.weighted_events:.2f}")
    print(f"Root holds      : {summary.root_local_events} events")
    print(f"Mass mean       : {summary.weighted_mass_mean:.4f} GeV")
    print(f"Mass std dev    : {summary.weighted_mass_std:.4f} GeV")
    print(f"Mass range      : [{summary.min_mass:.4f}, {summary.max_mass:.4f}] GeV")

    if using_mpi:
        print(
            "\nTip: re-run with a different number of ranks to watch the "
            "per-rank event counts change while the global metrics stay stable."
        )
    else:
        print(
            "\nTip: launch the script with 'mpirun -n 4' to see Laddu "
            "distribute the workload automatically."
        )


def _call_mpi(func: callable, *, default):
    try:
        value = func()
    except ModuleNotFoundError:  # pragma: no cover - triggered on non-MPI builds
        return default
    return default if value is None else value


if __name__ == "__main__":
    main()
