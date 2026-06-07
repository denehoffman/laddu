from typing import Any

from laddu.quantum import JLike, MLike

class Histogram:
    bin_edges: Any
    counts: Any
    total_weight: float

    def __init__(self, bin_edges: Any, counts: Any) -> None: ...
    @staticmethod
    def from_numpy(bin_edges: Any, counts: Any) -> Histogram: ...
    def to_numpy(self) -> tuple[Any, Any]: ...

def clebsch_gordan(
    j1: JLike, m1: MLike, j2: JLike, m2: MLike, j: JLike, m: MLike
) -> float: ...
