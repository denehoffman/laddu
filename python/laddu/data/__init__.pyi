import numpy as np
import numpy.typing as npt

from laddu.utils.variables import Mass
from laddu.utils.vectors import Vector3, Vector4

class Event:
    p4s: list[Vector4]
    eps: list[Vector3]
    weight: float
    def __init__(self, p4s: list[Vector4], eps: list[Vector3], weight: float): ...

class Dataset:
    events: list[Event]
    weights: npt.NDArray[np.float64]
    def __init__(self, events: list[Event]): ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Event: ...
    def len(self) -> int: ...
    def weighted_len(self) -> float: ...
    def bootstrap(self, seed: int) -> Dataset: ...

class BinnedDataset:
    bins: int
    range: tuple[float, float]
    edges: npt.NDArray[np.float64]
    def __len__(self) -> int: ...
    def len(self) -> int: ...
    def __getitem__(self, index: int) -> Dataset: ...

def open(path: str) -> Dataset: ...  # noqa: A001
def open_binned(
    path: str,
    variable: Mass,
    bins: int,
    range: tuple[float, float],  # noqa: A002
) -> BinnedDataset: ...
