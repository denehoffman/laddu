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
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import uproot

POLARIZATION_DEFAULT_MAG_NAME = "pol_magnitude"
POLARIZATION_DEFAULT_ANGLE_NAME = "pol_angle"


@dataclass
class AmpToolsData:
    beam_px: np.ndarray
    beam_py: np.ndarray
    beam_pz: np.ndarray
    beam_e: np.ndarray
    finals_px: np.ndarray
    finals_py: np.ndarray
    finals_pz: np.ndarray
    finals_e: np.ndarray
    weights: np.ndarray
    pol_magnitude: np.ndarray | None
    pol_angle: np.ndarray | None


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


def _read_scalar(
    tree: uproot.behaviors.TBranch.TBranch, *, entry_stop: int | None = None
) -> np.ndarray:
    array = tree.array(library="np", entry_stop=entry_stop)
    return np.asarray(array, dtype=np.float32)


def _read_matrix(
    tree: uproot.behaviors.TBranch.TBranch, *, entry_stop: int | None = None
) -> np.ndarray:
    # uproot returns an object array of per-event vectors; convert to a dense matrix
    raw = tree.array(library="np", entry_stop=entry_stop)
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


def _validate_final_shapes(
    e_final: np.ndarray,
    px_final: np.ndarray,
    py_final: np.ndarray,
    pz_final: np.ndarray,
) -> tuple[int, int]:
    n_events, n_finals = e_final.shape
    if not (px_final.shape == py_final.shape == pz_final.shape == (n_events, n_finals)):
        raise ValueError("Final-state branches must have a consistent shape")
    return n_events, n_finals


def _load_amptools_arrays(
    input_path: Path,
    tree_name: str,
    *,
    entry_stop: int | None = None,
) -> tuple[np.ndarray, ...]:
    with uproot.open(input_path) as file:
        try:
            tree = file[tree_name]
        except uproot.exceptions.KeyInFileError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Input file must contain a tree named '{tree_name}'") from exc

        e_beam = _read_scalar(tree["E_Beam"], entry_stop=entry_stop)
        px_beam = _read_scalar(tree["Px_Beam"], entry_stop=entry_stop)
        py_beam = _read_scalar(tree["Py_Beam"], entry_stop=entry_stop)
        pz_beam = _read_scalar(tree["Pz_Beam"], entry_stop=entry_stop)

        e_final = _read_matrix(tree["E_FinalState"], entry_stop=entry_stop)
        px_final = _read_matrix(tree["Px_FinalState"], entry_stop=entry_stop)
        py_final = _read_matrix(tree["Py_FinalState"], entry_stop=entry_stop)
        pz_final = _read_matrix(tree["Pz_FinalState"], entry_stop=entry_stop)

        if "Weight" in tree.keys():
            weight = _read_scalar(tree["Weight"], entry_stop=entry_stop)
        else:
            weight = np.ones_like(e_beam, dtype=np.float32)

    return e_beam, px_beam, py_beam, pz_beam, e_final, px_final, py_final, pz_final, weight


def _derive_polarization(
    px_beam: np.ndarray,
    py_beam: np.ndarray,
    *,
    pol_in_beam: bool,
    pol_angle_rad: float | None,
    pol_magnitude: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    beam_px = px_beam.copy()
    beam_py = py_beam.copy()
    pol_magnitude_arr: np.ndarray | None = None
    pol_angle_arr: np.ndarray | None = None

    if pol_in_beam:
        transverse_sq = px_beam.astype(np.float64) ** 2 + py_beam.astype(np.float64) ** 2
        pol_magnitude_arr = np.sqrt(transverse_sq).astype(np.float32)
        pol_angle_arr = np.arctan2(py_beam.astype(np.float64), px_beam.astype(np.float64)).astype(
            np.float32
        )
        beam_px.fill(0.0)
        beam_py.fill(0.0)
    elif pol_angle_rad is not None and pol_magnitude is not None:
        n_events = px_beam.shape[0]
        pol_magnitude_arr = np.full(n_events, pol_magnitude, dtype=np.float32)
        pol_angle_arr = np.full(n_events, pol_angle_rad, dtype=np.float32)

    return beam_px, beam_py, pol_magnitude_arr, pol_angle_arr


def _prepare_amptools_data(
    e_beam: np.ndarray,
    px_beam: np.ndarray,
    py_beam: np.ndarray,
    pz_beam: np.ndarray,
    e_final: np.ndarray,
    px_final: np.ndarray,
    py_final: np.ndarray,
    pz_final: np.ndarray,
    weight: np.ndarray,
    *,
    pol_in_beam: bool,
    pol_angle_rad: float | None,
    pol_magnitude: float | None,
) -> AmpToolsData:
    n_events, n_finals = _validate_final_shapes(e_final, px_final, py_final, pz_final)

    beam_px, beam_py, pol_magnitude_arr, pol_angle_arr = _derive_polarization(
        px_beam,
        py_beam,
        pol_in_beam=pol_in_beam,
        pol_angle_rad=pol_angle_rad,
        pol_magnitude=pol_magnitude,
    )

    return AmpToolsData(
        beam_px=beam_px,
        beam_py=beam_py,
        beam_pz=pz_beam,
        beam_e=e_beam,
        finals_px=px_final,
        finals_py=py_final,
        finals_pz=pz_final,
        finals_e=e_final,
        weights=weight.astype(np.float32),
        pol_magnitude=pol_magnitude_arr,
        pol_angle=pol_angle_arr,
    )


def read_root_file(
    path: str | Path,
    tree: str = "kin",
    *,
    pol_in_beam: bool = False,
    pol_angle_rad: float | None = None,
    pol_magnitude: float | None = None,
    num_entries: int | None = None,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[float]]:
    if pol_in_beam and (pol_angle_rad is not None or pol_magnitude is not None):
        raise ValueError("pol_in_beam cannot be combined with pol_angle_rad/pol_magnitude")
    if (pol_angle_rad is None) ^ (pol_magnitude is None):
        raise ValueError("pol_angle_rad and pol_magnitude must be provided together")

    input_path = Path(path)
    arrays = _load_amptools_arrays(input_path, tree, entry_stop=num_entries)
    data = _prepare_amptools_data(
        *arrays,
        pol_in_beam=pol_in_beam,
        pol_angle_rad=pol_angle_rad,
        pol_magnitude=pol_magnitude,
    )

    n_events, n_finals = data.finals_e.shape

    p4s_list: list[list[np.ndarray]] = []
    for event_idx in range(n_events):
        event_vectors: list[np.ndarray] = [
            np.array(
                [
                    data.beam_px[event_idx],
                    data.beam_py[event_idx],
                    data.beam_pz[event_idx],
                    data.beam_e[event_idx],
                ],
                dtype=np.float32,
            )
        ]
        for final_idx in range(n_finals):
            event_vectors.append(
                np.array(
                    [
                        data.finals_px[event_idx, final_idx],
                        data.finals_py[event_idx, final_idx],
                        data.finals_pz[event_idx, final_idx],
                        data.finals_e[event_idx, final_idx],
                    ],
                    dtype=np.float32,
                )
            )
        p4s_list.append(event_vectors)

    if data.pol_magnitude is not None and data.pol_angle is not None:
        polarisation_values = np.column_stack((data.pol_magnitude, data.pol_angle))
        eps_list = [[polarisation_values[i]] for i in range(n_events)]
    else:
        eps_list = [[] for _ in range(n_events)]

    weight_list = data.weights.tolist()

    return p4s_list, eps_list, weight_list


def convert(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    arrays = _load_amptools_arrays(input_path, "kin")
    pol_angle_rad = np.deg2rad(args.pol_angle) if args.pol_angle is not None else None
    data = _prepare_amptools_data(
        *arrays,
        pol_in_beam=args.pol_in_beam,
        pol_angle_rad=pol_angle_rad,
        pol_magnitude=args.pol_magnitude,
    )

    n_events, n_finals = data.finals_e.shape
    final_names = _ensure_names(args.p4_names, n_finals)

    columns: dict[str, np.ndarray] = {
        "beam_px": data.beam_px,
        "beam_py": data.beam_py,
        "beam_pz": data.beam_pz,
        "beam_e": data.beam_e,
    }

    for idx, name in enumerate(final_names):
        columns[f"{name}_px"] = data.finals_px[:, idx]
        columns[f"{name}_py"] = data.finals_py[:, idx]
        columns[f"{name}_pz"] = data.finals_pz[:, idx]
        columns[f"{name}_e"] = data.finals_e[:, idx]

    if data.pol_magnitude is not None and data.pol_angle is not None:
        columns[args.pol_magnitude_name] = data.pol_magnitude
        columns[args.pol_angle_name] = data.pol_angle

    columns["weight"] = data.weights

    pl.DataFrame({key: pl.Series(value) for key, value in columns.items()}).write_parquet(
        output_path,
        compression="zstd",
    )


def convert_from_amptools(
    input_path: str | Path,
    output_path: str | Path,
    *,
    p4_names: Sequence[str] | None = None,
    pol_in_beam: bool = False,
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    pol_angle_name: str = POLARIZATION_DEFAULT_ANGLE_NAME,
    pol_magnitude_name: str = POLARIZATION_DEFAULT_MAG_NAME,
) -> None:
    if pol_in_beam and (pol_angle is not None or pol_magnitude is not None):
        raise ValueError("pol_in_beam cannot be combined with pol_angle/pol_magnitude")
    if (pol_angle is None) ^ (pol_magnitude is None):
        raise ValueError("pol_angle and pol_magnitude must be provided together")

    input_path = Path(input_path)
    output_path = Path(output_path)

    args = argparse.Namespace(
        input=input_path,
        output=output_path,
        p4_names=list(p4_names) if p4_names is not None else None,
        pol_in_beam=pol_in_beam,
        pol_angle=pol_angle,
        pol_magnitude=pol_magnitude,
        pol_angle_name=pol_angle_name,
        pol_magnitude_name=pol_magnitude_name,
    )

    convert(input_path, output_path, args)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    convert(args.input, args.output, args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
