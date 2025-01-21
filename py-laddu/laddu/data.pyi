from pathlib import Path

import numpy as np
import numpy.typing as npt

from laddu.utils.variables import CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude
from laddu.utils.vectors import Vector3, Vector4

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
    p4s: list[Vector4]
    eps: list[Vector3]
    weight: float
    def __init__(self, p4s: list[Vector4], eps: list[Vector3], weight: float) -> None: ...
    def get_p4_sum(self, indices: list[int]) -> Vector4: ...

class Dataset:
    events: list[Event]
    weights: npt.NDArray[np.float64]
    def __init__(self, events: list[Event]) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Event: ...
    def len(self) -> int: ...
    def weighted_len(self) -> float: ...
    def bin_by(
        self,
        variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
        bins: int,
        range: tuple[float, float],
    ) -> BinnedDataset: ...
    def bootstrap(self, seed: int) -> Dataset: ...

class BinnedDataset:
    bins: int
    range: tuple[float, float]
    edges: npt.NDArray[np.float64]
    def __len__(self) -> int: ...
    def len(self) -> int: ...
    def __getitem__(self, index: int) -> Dataset: ...

def open(path: str) -> Dataset: ...
