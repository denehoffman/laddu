# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false
"""AmpTools → laddu Parquet converter.

This utility reads an AmpTools-style ROOT file and emits a Parquet dataset
compatible with ``laddu``'s named-column format.  The input file must contain a
single tree named ``kin`` with the following branches (all ``float32``):

``E_Beam``
    Scalar beam energy.
``Px_Beam``/``Py_Beam``/``Pz_Beam``
    Beam momentum components.
``E_FinalState``/``Px_FinalState``/``Py_FinalState``/``Pz_FinalState``
    Fixed-length arrays describing the final-state four-momenta.
``Weight`` (optional)
    Event weights.  When absent, events are assumed to be unweighted.

The converter always names the beam ``beam``.  Final-state particle names may be
supplied explicitly via ``--p4-names``; otherwise they default to ``particle1``,
``particle2`` … in the order provided by AmpTools.

Polarization information can be supplied in one of two ways:
* ``--pol-in-beam`` interprets the beam's transverse momentum as the
  polarization vector and zeros the exported ``beam_px``/``beam_py`` columns.
* ``--pol-angle`` (degrees) and ``--pol-magnitude`` apply constant values to all
  events.  Angles are stored in radians.

Polarization columns default to ``pol_magnitude`` and ``pol_angle`` when either
mode is enabled.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import uproot

POLARIZATION_DEFAULT_MAG_NAME = "pol_magnitude"
POLARIZATION_DEFAULT_ANGLE_NAME = "pol_angle"


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AmpTools ROOT data to laddu Parquet")
    parser.add_argument("input", type=Path, help="AmpTools ROOT file")
    parser.add_argument("output", type=Path, help="Destination Parquet file")
    parser.add_argument(
        "--p4-names",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Final-state particle names in AmpTools order (beam is always 'beam')",
    )
    parser.add_argument(
        "--pol-in-beam",
        action="store_true",
        help="Derive polarization from Px_Beam/Py_Beam and zero the exported transverse momentum",
    )
    parser.add_argument(
        "--pol-angle",
        type=float,
        default=None,
        help="Constant polarization angle in degrees (requires --pol-magnitude)",
    )
    parser.add_argument(
        "--pol-magnitude",
        type=float,
        default=None,
        help="Constant polarization magnitude (requires --pol-angle)",
    )
    parser.add_argument(
        "--pol-angle-name",
        default=POLARIZATION_DEFAULT_ANGLE_NAME,
        help="Name of the polarization angle column (default: pol_angle)",
    )
    parser.add_argument(
        "--pol-magnitude-name",
        default=POLARIZATION_DEFAULT_MAG_NAME,
        help="Name of the polarization magnitude column (default: pol_magnitude)",
    )

    args = parser.parse_args(argv)

    if args.pol_in_beam and (args.pol_angle is not None or args.pol_magnitude is not None):
        parser.error("--pol-in-beam cannot be combined with --pol-angle/--pol-magnitude")

    if (args.pol_angle is None) ^ (args.pol_magnitude is None):
        parser.error("--pol-angle and --pol-magnitude must be provided together")

    if (args.pol_angle_name != POLARIZATION_DEFAULT_ANGLE_NAME or args.pol_magnitude_name != POLARIZATION_DEFAULT_MAG_NAME) and not (
        args.pol_in_beam or args.pol_angle is not None
    ):
        parser.error("Polarization column names may only be customised when polarization data is exported")

    return args


def _read_scalar(tree: uproot.behaviors.TBranch.TBranch) -> np.ndarray:
    array = tree.array(library="np")
    return np.asarray(array, dtype=np.float32)


def _read_matrix(tree: uproot.behaviors.TBranch.TBranch) -> np.ndarray:
    # uproot returns an object array of per-event vectors; convert to a dense matrix
    raw = tree.array(library="np")
    matrix = np.asarray(list(raw), dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Final-state branches must be fixed-length arrays")
    return matrix


def _default_final_names(count: int) -> list[str]:
    return [f"particle{i}" for i in range(1, count + 1)]


def _ensure_names(names: Sequence[str] | None, count: int) -> list[str]:
    if names is None or len(names) == 0:
        return _default_final_names(count)
    if len(names) != count:
        raise ValueError(
            f"Expected {count} final-state names, received {len(names)}"
        )
    return list(names)


def convert(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    with uproot.open(input_path) as file:
        try:
            tree = file["kin"]
        except uproot.exceptions.KeyInFileError as exc:  # pragma: no cover - defensive
            raise KeyError("Input file must contain a tree named 'kin'") from exc

        e_beam = _read_scalar(tree["E_Beam"])
        px_beam = _read_scalar(tree["Px_Beam"])
        py_beam = _read_scalar(tree["Py_Beam"])
        pz_beam = _read_scalar(tree["Pz_Beam"])

        e_final = _read_matrix(tree["E_FinalState"])
        px_final = _read_matrix(tree["Px_FinalState"])
        py_final = _read_matrix(tree["Py_FinalState"])
        pz_final = _read_matrix(tree["Pz_FinalState"])

        if "Weight" in tree.keys():
            weight = _read_scalar(tree["Weight"])
        else:
            weight = np.ones_like(e_beam, dtype=np.float32)

    n_events, n_finals = e_final.shape
    if not (px_final.shape == py_final.shape == pz_final.shape == (n_events, n_finals)):
        raise ValueError("Final-state branches must have a consistent shape")

    final_names = _ensure_names(args.p4_names, n_finals)

    # Prepare beam components (may be overwritten for pol_in_beam)
    beam_px = px_beam.copy()
    beam_py = py_beam.copy()

    pol_magnitude: np.ndarray | None = None
    pol_angle: np.ndarray | None = None
    pol_mag_name = args.pol_magnitude_name
    pol_angle_name = args.pol_angle_name

    if args.pol_in_beam:
        transverse_sq = px_beam.astype(np.float64) ** 2 + py_beam.astype(np.float64) ** 2
        pol_magnitude = np.sqrt(transverse_sq).astype(np.float32)
        pol_angle = np.arctan2(py_beam.astype(np.float64), px_beam.astype(np.float64)).astype(
            np.float32
        )
        beam_px.fill(0.0)
        beam_py.fill(0.0)
    elif args.pol_angle is not None and args.pol_magnitude is not None:
        pol_magnitude = np.full(n_events, args.pol_magnitude, dtype=np.float32)
        pol_angle_value = np.deg2rad(args.pol_angle)
        pol_angle = np.full(n_events, pol_angle_value, dtype=np.float32)

    columns: dict[str, np.ndarray] = {
        "beam_px": beam_px,
        "beam_py": beam_py,
        "beam_pz": pz_beam,
        "beam_e": e_beam,
    }

    for idx, name in enumerate(final_names):
        columns[f"{name}_px"] = px_final[:, idx]
        columns[f"{name}_py"] = py_final[:, idx]
        columns[f"{name}_pz"] = pz_final[:, idx]
        columns[f"{name}_e"] = e_final[:, idx]

    if pol_magnitude is not None and pol_angle is not None:
        columns[pol_mag_name] = pol_magnitude
        columns[pol_angle_name] = pol_angle

    columns["weight"] = weight

    pl.DataFrame({key: pl.Series(value) for key, value in columns.items()}).write_parquet(
        output_path,
        compression="zstd",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    convert(args.input, args.output, args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
