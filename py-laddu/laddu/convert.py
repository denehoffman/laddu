# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false
"""
Usage:
    amptools-to-laddu <input_file> <output_file> [--tree <treename>] [--pol-in-beam | --pol-angle <angle> --pol-magnitude <magnitude>] [-n <num-entries>]

Options:
    --tree <treename>            The tree name in the ROOT file [default: kin].
    --pol-in-beam                Use the beam's momentum for polarization (eps).
    --pol-angle <angle>          The polarization angle in degrees (only used if --pol-in-beam is not used)
    --pol-magnitude <magnitude>  The polarization magnitude (only used if --pol-in-beam is not used)
    -n <num-entries>             Truncate the file to the first n entries for testing.
"""  # noqa: D205, D400

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa

import uproot
from docopt import docopt


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
    tree = uproot.open(f'{input_path}:{tree_name}')  #
    E_beam: npt.NDArray[np.float32] = tree['E_Beam'].array(
        library='np', entry_stop=num_entries
    )
    Px_beam: npt.NDArray[np.float32] = tree['Px_Beam'].array(
        library='np', entry_stop=num_entries
    )
    Py_beam: npt.NDArray[np.float32] = tree['Py_Beam'].array(
        library='np', entry_stop=num_entries
    )
    Pz_beam: npt.NDArray[np.float32] = tree['Pz_Beam'].array(
        library='np', entry_stop=num_entries
    )
    weight = (
        tree['Weight'].array(library='np', entry_stop=num_entries)
        if 'Weight' in tree
        else np.ones_like(E_beam)
    )

    E_final: npt.NDArray[np.float32] = np.array(
        list(tree['E_FinalState'].array(library='np', entry_stop=num_entries))
    )
    Px_final: npt.NDArray[np.float32] = np.array(
        list(tree['Px_FinalState'].array(library='np', entry_stop=num_entries))
    )
    Py_final: npt.NDArray[np.float32] = np.array(
        list(tree['Py_FinalState'].array(library='np', entry_stop=num_entries))
    )
    Pz_final: npt.NDArray[np.float32] = np.array(
        list(tree['Pz_FinalState'].array(library='np', entry_stop=num_entries))
    )

    p4_beam = (np.stack([Px_beam, Py_beam, Pz_beam, E_beam], axis=-1),)
    p4_final = np.stack([Px_final, Py_final, Pz_final, E_final], axis=-1)

    if 'EPS' in tree:
        eps = tree['EPS'].array(library='np', entry_stop=num_entries)
        eps = eps[:, np.newaxis, :]
    if 'eps' in tree:
        eps = tree['eps'].array(library='np', entry_stop=num_entries)
        eps = eps[:, np.newaxis, :]
    elif pol_in_beam:
        eps = np.stack([Px_beam, Py_beam, Pz_beam], axis=-1)[:, np.newaxis]
        p4_beam[:, 0] = 0  # Set Px to 0 # pyright: ignore[reportIndexIssue]
        p4_beam[:, 1] = 0  # Set Py to 0 # pyright: ignore[reportIndexIssue]
        p4_beam[:, 2] = E_beam  # Set Pz = E for beam # pyright: ignore[reportIndexIssue]
    elif pol_angle_rad is not None and pol_magnitude is not None:
        eps_x = pol_magnitude * np.cos(pol_angle_rad) * np.ones_like(E_beam)
        eps_y = pol_magnitude * np.sin(pol_angle_rad) * np.ones_like(E_beam)
        eps_z = np.zeros_like(E_beam)
        eps = np.stack([eps_x, eps_y, eps_z], axis=-1)[:, np.newaxis]
    else:
        eps = np.zeros((len(E_beam), 1, 3), dtype=np.float32)  # Default to 0

    p4s = np.concatenate([p4_beam[:, np.newaxis, :], p4_final], axis=1)  # pyright: ignore[reportCallIssue, reportArgumentType]

    return p4s.astype(np.float32), eps.astype(np.float32), weight


def save_as_parquet(
    p4s: npt.NDArray[np.float32],
    eps: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    output_path: Path | str,
) -> None:
    columns = {}
    n_particles = p4s.shape[1]
    for i in range(n_particles):
        columns[f'p4_{i}_Px'] = p4s[:, i, 0]
        columns[f'p4_{i}_Py'] = p4s[:, i, 1]
        columns[f'p4_{i}_Pz'] = p4s[:, i, 2]
        columns[f'p4_{i}_E'] = p4s[:, i, 3]

    n_eps = eps.shape[1]
    for i in range(n_eps):
        columns[f'aux_{i}_x'] = eps[:, i, 0]
        columns[f'aux_{i}_y'] = eps[:, i, 1]
        columns[f'aux_{i}_z'] = eps[:, i, 2]

    columns['weight'] = weight

    data = pa.table(columns)
    pa.write_table(data, str(output_path))


def convert_from_amptools(
    input_path: Path,
    output_path: Path,
    tree_name: str = 'kin',
    *,
    pol_in_beam: bool = False,
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    num_entries: int | None = None,
) -> None:
    p4s, eps, weight = read_root_file(
        input_path,
        tree_name,
        pol_in_beam=pol_in_beam,
        pol_angle_rad=pol_angle,
        pol_magnitude=pol_magnitude,
        num_entries=num_entries,
    )
    save_as_parquet(p4s, eps, weight, output_path)


def run() -> None:
    args = docopt(__doc__ if __doc__ else '')
    input_file = args['<input_file>']
    output_file = args['<output_file>']
    tree_name = args['--tree']
    pol_in_beam = args['--pol-in-beam']
    pol_angle = float(args['--pol-angle']) * np.pi / 180 if args['--pol-angle'] else None
    pol_magnitude = float(args['--pol-magnitude']) if args['--pol-magnitude'] else None
    num_entries = int(args['-n']) if args['-n'] else None

    convert_from_amptools(
        Path(input_file),
        Path(output_file),
        tree_name,
        pol_in_beam=pol_in_beam,
        pol_angle=pol_angle,
        pol_magnitude=pol_magnitude,
        num_entries=num_entries,
    )
