import numpy as np
import numpy.typing as npt

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

def open(path: str): ...  # noqa: A001
