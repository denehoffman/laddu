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
    """In-memory event container backed by :class:`laddu.laddu.Event` objects.

    The helper constructors accept ``dict`` objects, pandas/Polars frames, or
    numpy arrays and ensure that the expected four-momentum columns (``*_px``,
    ``*_py``, ``*_pz``, ``*_e``) are present.

    Examples
    --------
    >>> columns = {
    ...     'beam_px': [0.0], 'beam_py': [0.0], 'beam_pz': [9.0], 'beam_e': [9.5],
    ...     'k_pi_px': [0.2], 'k_pi_py': [0.1], 'k_pi_pz': [0.3], 'k_pi_e': [0.6],
    ...     'weight': [1.0], 'pol_magnitude': [0.5], 'pol_angle': [0.0],
    ... }
    >>> dataset = Dataset.from_dict(columns)
    >>> len(dataset)
    1
    """

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
                        raise KeyError(
                            f"Missing components {missing} for four-momentum '{base}'"
                        )
                    p4_names.append(base)
        if not p4_names:
            raise ValueError(
                'No four-momentum columns found (expected *_px, *_py, *_pz, *_e)'
            )
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
    def from_dict(
        data: dict[str, Any], rest_frame_of: list[str] | None = None
    ) -> Dataset:
        """Create a dataset from iterables keyed by column name.

        Parameters
        ----------
        data:
            Mapping whose keys are column names (e.g. ``beam_px``) and values are
            indexable sequences.
        rest_frame_of:
            Optional list of particle names whose combined rest frame should be
            used to boost each event (useful for quasi-two-body systems).
        """
        columns = {name: np.asarray(values) for name, values in data.items()}
        p4_names = Dataset._infer_p4_names(columns)
        component_names = {
            f'{name}_{suffix}' for name in p4_names for suffix in ('px', 'py', 'pz', 'e')
        }
        aux_names = Dataset._infer_aux_names(columns, component_names)

        n_events = len(columns[f'{p4_names[0]}_px'])
        weights = np.asarray(
            columns.get('weight', np.ones(n_events, dtype=float)), dtype=float
        )

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
    def from_numpy(
        data: dict[str, NDArray[np.floating]], rest_frame_of: list[str] | None = None
    ) -> Dataset:
        """Create a dataset from arrays without copying.

        Accepts any mapping of column names to ``ndarray`` objects and mirrors
        :meth:`from_dict`.
        """
        converted = {key: np.asarray(value) for key, value in data.items()}
        return Dataset.from_dict(converted, rest_frame_of=rest_frame_of)

    @staticmethod
    def from_pandas(
        data: pd.DataFrame, rest_frame_of: list[str] | None = None
    ) -> Dataset:
        """Materialise a dataset from a :class:`pandas.DataFrame`."""
        converted = {col: data[col].to_list() for col in data.columns}
        return Dataset.from_dict(converted, rest_frame_of=rest_frame_of)

    @staticmethod
    def from_polars(
        data: pl.DataFrame, rest_frame_of: list[str] | None = None
    ) -> Dataset:
        """Materialise a dataset from a :class:`polars.DataFrame`."""
        converted = {col: data[col].to_list() for col in data.columns}
        return Dataset.from_dict(converted, rest_frame_of=rest_frame_of)

    @staticmethod
    def open(
        path: str | Path,
        *,
        p4s: list[str] | None = None,
        aux: list[str] | None = None,
        boost_to_restframe_of: list[str] | None = None,
    ) -> Dataset:
        """Open a dataset from a file.

        Parameters
        ----------
        path:
            Parquet file on disk.
        p4s:
            Ordered list of particle base names (e.g. ``['beam', 'kshort1']``).
        aux:
            Auxiliary scalar columns to retain (such as ``pol_magnitude``).
        boost_to_restframe_of:
            Optional list of particle combinations used for rest-frame boosts.

        Notes
        -----
        Currently only supports ``.parquet`` files.
        """
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
    """Convert an AmpTools ROOT tuple directly into a :class:`Dataset`.

    Parameters
    ----------
    path:
        Input ROOT file.
    tree:
        Name of the TTree containing the kinematics (default ``'kin'``).
    pol_in_beam / pol_angle / pol_magnitude:
        Describe how to extract or override beam polarisation.
    num_entries:
        Limit the number of events read (useful for smoke tests).
    boost_to_com:
        When ``True``, events are boosted to the combined rest frame of all
        final-state particles.

    Examples
    --------
    >>> from laddu.data import open_amptools  # doctest: +SKIP
    >>> dataset = open_amptools('example_amp.root', pol_in_beam=True)  # doctest: +SKIP

    Notes
    -----
    This helper mirrors the CLI utility ``amptools-to-laddu``. It is handy in
    notebooks where shelling out would be cumbersome.
    """
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
