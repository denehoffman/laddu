from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from laddu.convert import read_root_file
from laddu.laddu import BinnedDataset, Dataset, Event, open
from laddu.utils.vectors import Vec3, Vec4

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
    boost_to_com: bool = True,
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
    n_particles = len(p4s_list[0])
    ds = Dataset(
        [
            Event(
                [Vec4.from_array(p4) for p4 in p4s],
                [Vec3.from_array(eps_vec) for eps_vec in eps],
                weight,
            )
            for p4s, eps, weight in zip(p4s_list, eps_list, weight_list)
        ]
    )
    if boost_to_com:
        return ds.boost_to_rest_frame_of(list(range(1, n_particles)))  # skip the beam
    return ds


__all__ = ['BinnedDataset', 'Dataset', 'Event', 'open', 'open_amptools']
