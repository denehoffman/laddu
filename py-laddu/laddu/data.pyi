from pathlib import Path
from typing import Any, overload

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray

from laddu.utils.variables import CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude
from laddu.utils.vectors import Vec3, Vec4

def open_amptools(
    path: str | Path,
    tree: str = 'kin',
    *,
    pol_in_beam: bool = False,
    pol_angle: float | None = None,
    pol_magnitude: float | None = None,
    num_entries: int | None = None,
    boost_to_com: bool = True,
) -> Dataset: ...

class Event:
    p4s: list[Vec4]
    aux: list[Vec3]
    weight: float
    def __init__(
        self,
        p4s: list[Vec4],
        aux: list[Vec3],
        weight: float,
        *,
        rest_frame_indices: list[int] | None = None,
    ) -> None: ...
    def get_p4_sum(self, indices: list[int]) -> Vec4: ...
    def boost_to_rest_frame_of(self, indices: list[int]) -> Event: ...
    def evaluate(
        self, variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam
    ) -> float: ...

class Dataset:
    events: list[Event]
    n_events: int
    n_events_weighted: float
    weights: NDArray[np.float64]
    def __init__(self, events: list[Event]) -> None: ...
    def __len__(self) -> int: ...
    def __add__(self, other: Dataset | int) -> Dataset: ...
    def __radd__(self, other: Dataset | int) -> Dataset: ...
    @overload
    def __getitem__(self, index: int) -> Event: ...
    @overload
    def __getitem__(
        self, index: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam
    ) -> NDArray[np.float64]: ...
    def __getitem__(
        self, index: int | Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam
    ) -> Event | NDArray[np.float64]: ...
    def bin_by(
        self,
        variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
        bins: int,
        range: tuple[float, float],
    ) -> BinnedDataset: ...
    def bootstrap(self, seed: int) -> Dataset: ...
    def boost_to_rest_frame_of(self, indices: list[int]) -> Dataset: ...
    @staticmethod
    def from_dict(
        data: dict[str, Any], rest_frame_indices: list[int] | None = None
    ) -> Dataset: ...
    @staticmethod
    def from_numpy(
        data: dict[str, NDArray[np.floating]], rest_frame_indices: list[int] | None = None
    ) -> Dataset: ...
    @staticmethod
    def from_pandas(
        data: pd.DataFrame, rest_frame_indices: list[int] | None = None
    ) -> Dataset: ...
    @staticmethod
    def from_polars(
        data: pl.DataFrame, rest_frame_indices: list[int] | None = None
    ) -> Dataset: ...
    def evaluate(
        self, variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam
    ) -> NDArray[np.float64]: ...

class BinnedDataset:
    n_bins: int
    range: tuple[float, float]
    edges: NDArray[np.float64]
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Dataset: ...

def open(path: str | Path, *, rest_frame_indices: list[int] | None = None) -> Dataset: ...
