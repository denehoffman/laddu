from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from laddu.convert import read_root_file
from laddu.laddu import BinnedDataset, DatasetBase, Event, open
from laddu.utils.vectors import Vec3, Vec4

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from numpy.typing import NDArray


class Dataset(DatasetBase):
    P4_PREFIX: str = 'p4_'
    AUX_PREFIX: str = 'aux_'

    @staticmethod
    def from_dict(
        data: dict[str, Any], rest_frame_indices: list[int] | None = None
    ) -> Dataset:
        """
        Create a Dataset from a dict.

        Arguments
        ---------
        data: dict of str to Any
            dict of lists with keys matching the dataset format
        rest_frame_indices: list of int, optional
            If provided, the dataset will be boosted to the rest frame
            of the 4-momenta specified by the indices

        Returns
        -------
        Dataset
        """
        p4_count = len([key for key in data if key.startswith(Dataset.P4_PREFIX)]) // 4
        aux_count = len([key for key in data if key.startswith(Dataset.AUX_PREFIX)]) // 3
        events = []
        n_events = len(data[f'{Dataset.P4_PREFIX}0_Px'])
        weights = data.get('weight', np.ones(n_events))
        for i in range(n_events):
            p4s = [
                Vec4.from_array(
                    [
                        data[f'{Dataset.P4_PREFIX}{j}_Px'][i],
                        data[f'{Dataset.P4_PREFIX}{j}_Py'][i],
                        data[f'{Dataset.P4_PREFIX}{j}_Pz'][i],
                        data[f'{Dataset.P4_PREFIX}{j}_E'][i],
                    ]
                )
                for j in range(p4_count)
            ]
            aux = [
                Vec3.from_array(
                    [
                        data[f'{Dataset.AUX_PREFIX}{j}_x'][i],
                        data[f'{Dataset.AUX_PREFIX}{j}_y'][i],
                        data[f'{Dataset.AUX_PREFIX}{j}_z'][i],
                    ]
                )
                for j in range(aux_count)
            ]
            weight = weights[i]
            events.append(Event(p4s, aux, weight, rest_frame_indices=rest_frame_indices))
        return DatasetBase(events)

    @staticmethod
    def from_numpy(
        data: dict[str, NDArray[np.floating]], rest_frame_indices: list[int] | None = None
    ) -> Dataset:
        """
        Create a Dataset from a dict of numpy arrays.

        Arguments
        ---------
        data: dict of str to NDArray
            dict of arrays with keys matching the dataset format
        rest_frame_indices: list of int, optional
            If provided, the dataset will be boosted to the rest frame
            of the 4-momenta specified by the indices

        Returns
        -------
        Dataset
        """
        return Dataset.from_dict(data, rest_frame_indices=rest_frame_indices)

    @staticmethod
    def from_pandas(
        data: pd.DataFrame, rest_frame_indices: list[int] | None = None
    ) -> Dataset:
        """
        Create a Dataset from a pandas DataFrame.

        Arguments
        ---------
        data: pandas.DataFrame
            DataFrame with columns matching the dataset format
        rest_frame_indices: list of int, optional
            If provided, the dataset will be boosted to the rest frame
            of the 4-momenta specified by the indices

        Returns
        -------
        Dataset
        """
        return Dataset.from_dict(data.to_dict(), rest_frame_indices=rest_frame_indices)

    @staticmethod
    def from_polars(
        data: pl.DataFrame, rest_frame_indices: list[int] | None = None
    ) -> Dataset:
        """
        Create a Dataset from a polars DataFrame.

        Arguments
        ---------
        data: polars.DataFrame
            DataFrame with columns matching the dataset format
        rest_frame_indices: list of int, optional
            If provided, the dataset will be boosted to the rest frame
            of the 4-momenta specified by the indices

        Returns
        -------
        Dataset
        """
        return Dataset.from_dict(data.to_dict(), rest_frame_indices=rest_frame_indices)

    @staticmethod
    def from_arrow(
        data: pa.Table, rest_frame_indices: list[int] | None = None
    ) -> Dataset:
        """
        Create a Dataset from a pyarrow Table.

        Arguments
        ---------
        data: pyarrow.Table
            Table with columns matching the dataset format
        rest_frame_indices: list of int, optional
            If provided, the dataset will be boosted to the rest frame
            of the 4-momenta specified by the indices

        Returns
        -------
        Dataset
        """
        return Dataset.from_dict(data.to_pydict(), rest_frame_indices=rest_frame_indices)


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
    boost_indices = list(range(1, n_particles)) if boost_to_com else None
    ds = Dataset(
        [
            Event(
                [Vec4.from_array(p4) for p4 in p4s],
                [Vec3.from_array(eps_vec) for eps_vec in eps],
                weight,
                boost_indices=boost_indices,
            )
            for p4s, eps, weight in zip(p4s_list, eps_list, weight_list)
        ]
    )
    return ds


__all__ = ['BinnedDataset', 'Dataset', 'Event', 'open', 'open_amptools']
