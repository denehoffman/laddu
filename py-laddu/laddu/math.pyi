from typing import Any

class Histogram:
    bin_edges: Any
    counts: Any
    total_weight: float

    def __init__(self, bin_edges: Any, counts: Any) -> None: ...
    @staticmethod
    def from_numpy(bin_edges: Any, counts: Any) -> Histogram: ...
    def to_numpy(self) -> tuple[Any, Any]: ...
