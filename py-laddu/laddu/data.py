from __future__ import annotations as _annotations

from typing import Any as _Any
from typing import Literal as _Literal
from typing import cast as _cast

from laddu.laddu import BinnedDataset, Dataset, Event


def _dataset_to_arrow(
    self: Dataset, *, precision: _Literal['f64', 'f32'] = 'f64'
) -> _Any:
    from . import io as _io

    return _io.to_arrow(self, precision=precision)


# PyO3 exposes Dataset as a Python type; attach the convenience method on the
# Python side so Arrow support can reuse the existing io-layer column export.
_cast(_Any, Dataset).to_arrow = _dataset_to_arrow

__all__ = [
    'BinnedDataset',
    'Dataset',
    'Event',
]
