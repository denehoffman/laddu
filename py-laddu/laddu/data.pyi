from pathlib import Path

import numpy as np
import numpy.typing as npt

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
) -> Dataset: ...

class Event:
    p4s: list[Vec4]
    eps: list[Vec3]
    weight: float
    def __init__(self, p4s: list[Vec4], eps: list[Vec3], weight: float) -> None: ...
    def get_p4_sum(self, indices: list[int]) -> Vec4: ...
    def boost_to_rest_frame_of(self, indices: list[int]) -> Event: ...

class Dataset:
    events: list[Event]
    n_events: int
    n_events_weighted: float
    weights: npt.NDArray[np.float64]
    def __init__(self, events: list[Event]) -> None: ...
    def __len__(self) -> int: ...
    def __add__(self, other: Dataset | int) -> Dataset: ...
    def __radd__(self, other: Dataset | int) -> Dataset: ...
    def __getitem__(self, index: int) -> Event: ...
    def bin_by(
        self,
        variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
        bins: int,
        range: tuple[float, float],
    ) -> BinnedDataset: ...
    def bootstrap(self, seed: int) -> Dataset: ...
    def boost_to_rest_frame_of(self, indices: list[int]) -> Dataset: ...

class BinnedDataset:
    n_bins: int
    range: tuple[float, float]
    edges: npt.NDArray[np.float64]
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Dataset: ...

def open(path: str) -> Dataset: ...
