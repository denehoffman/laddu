# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false
"""
Usage:
    amptools-to-laddu <input_file> <output_file> [--tree <treename>] [--pol-in-beam | --pol-angle <angle> --pol-magnitude <magnitude>] [--p4-names <name>...] [-n <num-entries>]

Options:
    --tree <treename>            The tree name in the ROOT file [default: kin].
    --pol-in-beam                Use the beam's momentum for polarization (eps).
    --pol-angle <angle>          The polarization angle in degrees (only used if --pol-in-beam is not used)
    --pol-magnitude <magnitude>  The polarization magnitude (only used if --pol-in-beam is not used)
    --p4-names <name>...         Particle names in the order provided by AmpTools (beam first).
    -n <num-entries>             Truncate the file to the first n entries for testing.
"""  # noqa: D205, D400

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl
import uproot
from docopt import docopt

_POLARIZATION_ZERO_TOL = 1e-9


def _stack_final_component(
    tree: uproot.ReadOnlyDirectory, field: str, num_entries: int | None
) -> npt.NDArray[np.float32]:
    return np.array(
        list(tree[field].array(library='np', entry_stop=num_entries)), dtype=np.float32
    )


def _detect_polarization_four_vector(
    p4_final: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32] | None, npt.NDArray[np.float32]]:
    if p4_final.size == 0 or p4_final.shape[1] == 0:
        return None, p4_final

    candidate = p4_final[:, -1, :]
    if np.allclose(candidate[:, 2], 0.0, atol=_POLARIZATION_ZERO_TOL) and np.allclose(
        candidate[:, 3], 0.0, atol=_POLARIZATION_ZERO_TOL
    ):
        return candidate[:, :3], p4_final[:, :-1, :]

    return None, p4_final


def _normalize_polarization_shape(
    values: npt.NDArray[np.float32] | None,
    length: int,
) -> npt.NDArray[np.float32]:
    if values is None:
        return np.zeros((length, 3), dtype=np.float32)

    if values.ndim == 3:
        if values.shape[1] != 1:
            msg = f'Unexpected polarization shape {values.shape}; expected (N, 1, 3).'
            raise ValueError(msg)
        values = values[:, 0, :]
    if values.shape != (length, 3):
        msg = f'Polarization array has shape {values.shape}, expected ({length}, 3).'
        raise ValueError(msg)
    return values.astype(np.float32, copy=False)


def read_root_file(
    input_path: Path | str,
    tree_name: str = 'kin',
    *,
    pol_in_beam: bool = False,
    pol_angle_rad: float | None = None,
    pol_magnitude: float | None = None,
    num_entries: int | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    input_path = Path(input_path)
    tree = uproot.open(f'{input_path}:{tree_name}')

    e_beam: npt.NDArray[np.float32] = tree['E_Beam'].array(
        library='np', entry_stop=num_entries
    )
    px_beam: npt.NDArray[np.float32] = tree['Px_Beam'].array(
        library='np', entry_stop=num_entries
    )
    py_beam: npt.NDArray[np.float32] = tree['Py_Beam'].array(
        library='np', entry_stop=num_entries
    )
    pz_beam: npt.NDArray[np.float32] = tree['Pz_Beam'].array(
        library='np', entry_stop=num_entries
    )
    weight = (
        tree['Weight'].array(library='np', entry_stop=num_entries)
        if 'Weight' in tree
        else np.ones_like(e_beam)
    )

    e_final = _stack_final_component(tree, 'E_FinalState', num_entries)
    px_final = _stack_final_component(tree, 'Px_FinalState', num_entries)
    py_final = _stack_final_component(tree, 'Py_FinalState', num_entries)
    pz_final = _stack_final_component(tree, 'Pz_FinalState', num_entries)

    p4_beam = np.stack([px_beam, py_beam, pz_beam, e_beam], axis=-1).astype(np.float32)
    p4_final = np.stack([px_final, py_final, pz_final, e_final], axis=-1).astype(np.float32)

    polarization_vectors: npt.NDArray[np.float32] | None
    if 'EPS' in tree:
        eps = tree['EPS'].array(library='np', entry_stop=num_entries)
        polarization_vectors = np.asarray(eps, dtype=np.float32)
    elif 'eps' in tree:
        eps = tree['eps'].array(library='np', entry_stop=num_entries)
        polarization_vectors = np.asarray(eps, dtype=np.float32)
    elif pol_in_beam:
        polarization_vectors = np.stack([px_beam, py_beam, pz_beam], axis=-1).astype(
            np.float32
        )
        p4_beam[:, 0] = 0  # Set Px to 0
        p4_beam[:, 1] = 0  # Set Py to 0
        p4_beam[:, 2] = e_beam  # Set Pz = E for beam
    elif pol_angle_rad is not None and pol_magnitude is not None:
        eps_x = pol_magnitude * np.cos(pol_angle_rad) * np.ones_like(e_beam)
        eps_y = pol_magnitude * np.sin(pol_angle_rad) * np.ones_like(e_beam)
        eps_z = np.zeros_like(e_beam)
        polarization_vectors = np.stack([eps_x, eps_y, eps_z], axis=-1).astype(np.float32)
    else:
        polarization_vectors = None

    if polarization_vectors is None:
        beam_placeholder_mag = np.sqrt(px_beam * px_beam + py_beam * py_beam)
        if np.any(beam_placeholder_mag > _POLARIZATION_ZERO_TOL):
            polarization_vectors = np.stack(
                [px_beam, py_beam, np.zeros_like(px_beam)], axis=-1
            ).astype(np.float32)
            p4_beam[:, 0] = 0.0
            p4_beam[:, 1] = 0.0

    derived_polarization, p4_final = _detect_polarization_four_vector(p4_final)
    if polarization_vectors is None and derived_polarization is not None:
        polarization_vectors = derived_polarization

    polarization_vectors = _normalize_polarization_shape(polarization_vectors, len(e_beam))

    p4s = np.concatenate([p4_beam[:, np.newaxis, :], p4_final], axis=1)

    return p4s.astype(np.float32), polarization_vectors, weight.astype(np.float32)


def _default_particle_names(count: int) -> list[str]:
    if count < 1:
        msg = 'Expected at least one four-momentum from AmpTools.'
        raise ValueError(msg)
    return ['beam', *[f'particle{i}' for i in range(1, count)]]


def save_as_parquet(
    p4s: npt.NDArray[np.float32],
    polarization_vectors: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    output_path: Path | str,
    particle_names: Sequence[str],
) -> None:
    if p4s.shape[1] != len(particle_names):
        msg = (
            f'Number of particle names ({len(particle_names)}) does not match '
            f'number of four-momenta ({p4s.shape[1]}).'
        )
        raise ValueError(msg)

    columns: dict[str, npt.NDArray[np.float32]] = {}
    for index, name in enumerate(particle_names):
        columns[f'{name}_px'] = p4s[:, index, 0]
        columns[f'{name}_py'] = p4s[:, index, 1]
        columns[f'{name}_pz'] = p4s[:, index, 2]
        columns[f'{name}_e'] = p4s[:, index, 3]

    pol_x = polarization_vectors[:, 0]
    pol_y = polarization_vectors[:, 1]
    pol_magnitude = np.sqrt(pol_x * pol_x + pol_y * pol_y).astype(np.float32)
    pol_angle = np.arctan2(pol_y, pol_x).astype(np.float32)

    columns['pol_magnitude'] = pol_magnitude
    columns['pol_angle'] = pol_angle
    columns['weight'] = weight

    dataframe = pl.DataFrame({name: pl.Series(value) for name, value in columns.items()})
    dataframe.write_parquet(str(output_path), compression='zstd')


def convert_from_amptools(
    input_path: Path,
    output_path: Path,
    tree_name: str = 'kin',
    *,
    pol_in_beam: bool = False,
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    num_entries: int | None = None,
    particle_names: Sequence[str] | None = None,
) -> None:
    p4s, polarization_vectors, weight = read_root_file(
        input_path,
        tree_name,
        pol_in_beam=pol_in_beam,
        pol_angle_rad=pol_angle,
        pol_magnitude=pol_magnitude,
        num_entries=num_entries,
    )

    names = list(particle_names) if particle_names else _default_particle_names(p4s.shape[1])
    save_as_parquet(p4s, polarization_vectors, weight, output_path, names)


def run() -> None:
    args = docopt(__doc__ if __doc__ else '')
    input_file = args['<input_file>']
    output_file = args['<output_file>']
    tree_name = args['--tree']
    pol_in_beam = args['--pol-in-beam']
    pol_angle = float(args['--pol-angle']) * np.pi / 180 if args['--pol-angle'] else None
    pol_magnitude = float(args['--pol-magnitude']) if args['--pol-magnitude'] else None
    num_entries = int(args['-n']) if args['-n'] else None
    particle_names = args['--p4-names']
    if particle_names is not None and len(particle_names) == 0:
        particle_names = None

    convert_from_amptools(
        Path(input_file),
        Path(output_file),
        tree_name,
        pol_in_beam=pol_in_beam,
        pol_angle=pol_angle,
        pol_magnitude=pol_magnitude,
        num_entries=num_entries,
        particle_names=particle_names,
    )
