"""
Usage:
    amptools-to-laddu <input_file> <output_file> [--tree <treename>] [--pol-in-beam | --pol-angle <angle> --pol-magnitude <magnitude>] [-n <num-entries>]
Options:
    --tree <treename>            The tree name in the ROOT file [default: kin].
    --pol-in-beam                Use the beam's momentum for polarization (eps).
    --pol-angle <angle>          The polarization angle in degrees (only used if --pol-in-beam is not used)
    --pol-magnitude <magnitude>  The polarization magnitude (only used if --pol-in-beam is not used)
    -n <num-entries>             Truncate the file to the first n entries for testing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from docopt import docopt
from loguru import logger


def read_root_file(
    input_file, tree_name, pol_in_beam, pol_angle, pol_magnitude, num_entries=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ROOT file and extract data with optional polarization from the beam."""
    logger.info(f"Reading ROOT file: {input_file}")
    tfile = uproot.open(input_file)
    tree = tfile[tree_name]
    logger.info(f"Using tree: {tree_name}")
    # Read necessary branches
    E_beam = tree["E_Beam"].array(library="np", entry_stop=num_entries)  # pyright: ignore
    Px_beam = tree["Px_Beam"].array(library="np", entry_stop=num_entries)  # pyright: ignore
    Py_beam = tree["Py_Beam"].array(library="np", entry_stop=num_entries)  # pyright: ignore
    Pz_beam = tree["Pz_Beam"].array(library="np", entry_stop=num_entries)  # pyright: ignore
    weight = tree["Weight"].array(library="np", entry_stop=num_entries) if "Weight" in tree else np.ones_like(E_beam)  # pyright: ignore

    # Final state particles
    E_final = np.array(list(tree["E_FinalState"].array(library="np", entry_stop=num_entries)))  # pyright: ignore
    Px_final = np.array(list(tree["Px_FinalState"].array(library="np", entry_stop=num_entries)))  # pyright: ignore
    Py_final = np.array(list(tree["Py_FinalState"].array(library="np", entry_stop=num_entries)))  # pyright: ignore
    Pz_final = np.array(list(tree["Pz_FinalState"].array(library="np", entry_stop=num_entries)))  # pyright: ignore

    # Handle beam four-vector: (nevents, 4)
    p4_beam = np.stack([Px_beam, Py_beam, Pz_beam, E_beam], axis=-1)

    # Handle final state four-vectors: (nevents, nparticles, 4)
    p4_final = np.stack([Px_final, Py_final, Pz_final, E_final], axis=-1)

    # Check if EPS branch exists and update eps if needed
    if "EPS" in tree:
        logger.info("EPS branch found. Using it for eps values.")
        eps = tree["EPS"].array(library="np", entry_stop=num_entries)  # pyright: ignore
        eps = eps[:, np.newaxis, :]
    if "eps" in tree:
        logger.info("eps branch found. Using it for eps values.")
        eps = tree["eps"].array(library="np", entry_stop=num_entries)  # pyright: ignore
        eps = eps[:, np.newaxis, :]
    elif pol_in_beam:
        logger.info("Using beam's momentum for polarization (eps).")
        eps = np.stack([Px_beam, Py_beam, Pz_beam], axis=-1)[:, np.newaxis]
        # Reset beam momentum
        p4_beam[:, 0] = 0  # Set Px to 0
        p4_beam[:, 1] = 0  # Set Py to 0
        p4_beam[:, 2] = E_beam  # Set Pz = E for beam
    elif pol_angle is not None and pol_magnitude is not None:
        logger.info(f"Using input polarization angle ({pol_angle}) and magnitude ({pol_magnitude}).")
        eps_x = pol_magnitude * np.cos(pol_angle) * np.ones_like(E_beam)
        eps_y = pol_magnitude * np.sin(pol_angle) * np.ones_like(E_beam)
        eps_z = np.zeros_like(E_beam)
        eps = np.stack([eps_x, eps_y, eps_z], axis=-1)[:, np.newaxis]
    else:
        logger.info("Using default or provided eps values.")
        eps = np.zeros((len(E_beam), 1, 3), dtype=np.float32)  # Default to 0

    # Concatenate the beam and final state particles: (nevents, nparticles+1, 4)
    logger.info("Concatenating beam and final state particles.")
    p4s = np.concatenate([p4_beam[:, np.newaxis, :], p4_final], axis=1)

    return p4s.astype(np.float32), eps.astype(np.float32), weight


def save_as_parquet(p4s, eps, weight, output_file):
    """Save the processed data into Parquet format."""
    logger.info("Saving data to Parquet format.")

    # Flatten the p4s and eps into individual columns
    columns = {}
    n_particles = p4s.shape[1]
    for i in range(n_particles):
        columns[f"p4_{i}_Px"] = p4s[:, i, 0]
        columns[f"p4_{i}_Py"] = p4s[:, i, 1]
        columns[f"p4_{i}_Pz"] = p4s[:, i, 2]
        columns[f"p4_{i}_E"] = p4s[:, i, 3]

    n_eps = eps.shape[1]
    for i in range(n_eps):
        columns[f"eps_{i}_x"] = eps[:, i, 0]
        columns[f"eps_{i}_y"] = eps[:, i, 1]
        columns[f"eps_{i}_z"] = eps[:, i, 2]

    # Add weights
    columns["weight"] = weight

    # Create a DataFrame and save as Parquet
    data = pd.DataFrame(columns)
    data.to_parquet(output_file, index=False)
    logger.info(f"File saved: {output_file}")


def convert_from_amptools(
    input_path: Path,
    output_path: Path,
    tree_name: str = "kin",
    pol_in_beam: bool = False,  # noqa: FBT001, FBT002
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    num_entries: int | None = None,
):
    p4s, eps, weight = read_root_file(input_path, tree_name, pol_in_beam, pol_angle, pol_magnitude, num_entries)
    save_as_parquet(p4s, eps, weight, output_path)


def run():
    """Main entry point for the script."""
    args = docopt(__doc__ if __doc__ else "")
    input_file = args["<input_file>"]
    output_file = args["<output_file>"]
    tree_name = args["--tree"]
    pol_in_beam = args["--pol-in-beam"]
    pol_angle = float(args["--pol-angle"]) * np.pi / 180 if args["--pol-angle"] else None
    pol_magnitude = float(args["--pol-magnitude"]) if args["--pol-magnitude"] else None
    num_entries = int(args["-n"]) if args["-n"] else None

    convert_from_amptools(
        Path(input_file), Path(output_file), tree_name, pol_in_beam, pol_angle, pol_magnitude, num_entries
    )

    df_read = pd.read_parquet(output_file)
    print("Output Parquet File (head):")  # noqa: T201
    print(df_read.head())  # noqa: T201
    print(f"Total Entries: {len(df_read)}")  # noqa: T201
