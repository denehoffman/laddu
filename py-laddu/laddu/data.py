from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from laddu.convert import read_root_file
from laddu.laddu import BinnedDataset, DatasetBase, Event
from laddu.utils.vectors import Vec4

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import polars as pl
    from numpy.typing import NDArray


class Dataset(DatasetBase):
    @staticmethod
    def _infer_p4_names(columns: dict[str, Any]) -> list[str]:
        if any(key.startswith('p4_') for key in columns):  # legacy format
            msg = 'Legacy column format detected (p4_N_*). Please run convert_legacy_parquet.py first.'
            raise ValueError(msg)
        p4_names: list[str] = []
        for key in columns:
            if key.endswith('_px'):
                base = key[:-3]
                if base not in p4_names:
                    required = [f'{base}_{suffix}' for suffix in ('px', 'py', 'pz', 'e')]
                    missing = [name for name in required if name not in columns]
                    if missing:
                        raise KeyError(f"Missing components {missing} for four-momentum '{base}'")
                    p4_names.append(base)
        if not p4_names:
            raise ValueError('No four-momentum columns found (expected *_px, *_py, *_pz, *_e)')
        return p4_names

    @staticmethod
    def _infer_aux_names(columns: dict[str, Any], used: set[str]) -> list[str]:
        aux_names: list[str] = []
        for key in columns:
            if key == 'weight' or key in used:
                continue
            aux_names.append(key)
        return aux_names

    @staticmethod
    def from_dict(data: dict[str, Any], rest_frame_of: list[str] | None = None) -> Dataset:
        """Create a Dataset from a dictionary mapping column names to sequences."""
        columns = {name: np.asarray(values) for name, values in data.items()}
        p4_names = Dataset._infer_p4_names(columns)
        component_names = {f'{name}_{suffix}' for name in p4_names for suffix in ('px', 'py', 'pz', 'e')}
        aux_names = Dataset._infer_aux_names(columns, component_names)

        n_events = len(columns[f'{p4_names[0]}_px'])
        weights = np.asarray(columns.get('weight', np.ones(n_events, dtype=float)), dtype=float)

        events: list[Event] = []
        for i in range(n_events):
            p4s = [
                Vec4.from_array(
                    [
                        float(columns[f'{name}_px'][i]),
                        float(columns[f'{name}_py'][i]),
                        float(columns[f'{name}_pz'][i]),
                        float(columns[f'{name}_e'][i]),
                    ]
                )
                for name in p4_names
            ]
            aux_values = [float(columns[name][i]) for name in aux_names]
            events.append(
                Event(
                    p4s,
                    aux_values,
                    float(weights[i]),
                    rest_frame_of=rest_frame_of,
                    p4_names=p4_names,
                    aux_names=aux_names,
                )
            )

        return Dataset(events, p4_names=p4_names, aux_names=aux_names)

    @staticmethod
    def from_numpy(data: dict[str, NDArray[np.floating]], rest_frame_of: list[str] | None = None) -> Dataset:
        converted = {key: np.asarray(value) for key, value in data.items()}
        return Dataset.from_dict(converted, rest_frame_of=rest_frame_of)

    @staticmethod
    def from_pandas(data: pd.DataFrame, rest_frame_of: list[str] | None = None) -> Dataset:
        converted = {col: data[col].to_list() for col in data.columns}
        return Dataset.from_dict(converted, rest_frame_of=rest_frame_of)

    @staticmethod
    def from_polars(data: pl.DataFrame, rest_frame_of: list[str] | None = None) -> Dataset:
        converted = {col: data[col].to_list() for col in data.columns}
        return Dataset.from_dict(converted, rest_frame_of=rest_frame_of)

    @staticmethod
    def open(
        path: str | Path,
        *,
        p4s: list[str],
        aux: list[str],
        boost_to_restframe_of: list[str] | None = None,
    ) -> Dataset:
        """Open a dataset from a Parquet file."""
        return DatasetBase.open(
            path,
            p4s=p4s,
            aux=aux,
            boost_to_restframe_of=boost_to_restframe_of,
        )


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
    p4_names = [f'particle{i}' for i in range(n_particles)]
    sample_eps = eps_list[0] if eps_list else []
    aux_len = sum(len(vec) for vec in sample_eps)
    aux_names = [f'aux_{i}' for i in range(aux_len)]
    rest_frame_of = p4_names[1:] if boost_to_com else None
    events = []
    for p4s, eps, weight in zip(p4s_list, eps_list, weight_list):
        p4_vectors = [Vec4.from_array(p4) for p4 in p4s]
        aux_values = [float(component) for eps_vec in eps for component in eps_vec]
        events.append(
            Event(
                p4_vectors,
                aux_values,
                weight,
                rest_frame_of=rest_frame_of,
                p4_names=p4_names,
                aux_names=aux_names,
            )
        )
    ds = Dataset(events, p4_names=p4_names, aux_names=aux_names)
    return ds


__all__ = ['BinnedDataset', 'Dataset', 'Event', 'open', 'open_amptools']
