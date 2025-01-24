from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from laddu.convert import read_root_file
from laddu.laddu import BinnedDataset, Dataset, Event, open
from laddu.utils.vectors import Vector3, Vector4

if TYPE_CHECKING:
    from pathlib import Path


def open_amptools(
    path: str | Path,
    tree: str = 'kin',
    *,
    pol_in_beam: bool = False,
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    num_entries: int | None = None,
) -> Dataset:
    pol_angle_rad = pol_angle * np.pi / 180 if pol_angle else None
    p4s_list, eps_list, weight_list = read_root_file(
        path,
        tree,
        pol_in_beam=pol_in_beam,
        pol_angle_rad=pol_angle_rad,
        pol_magnitude=pol_magnitude,
        num_entries=num_entries,
    )
    return Dataset(
        [
            Event(
                [Vector4.from_array(p4) for p4 in p4s],
                [Vector3.from_array(eps_vec) for eps_vec in eps],
                weight,
            )
            for p4s, eps, weight in zip(p4s_list, eps_list, weight_list)
        ]
    )


__all__ = ['BinnedDataset', 'Dataset', 'Event', 'open']
