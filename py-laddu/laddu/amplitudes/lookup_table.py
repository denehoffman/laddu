"""Lookup-table amplitudes over event variables.

The ``axis_coordinates`` argument is interpreted by interpolation mode:
``nearest`` uses bin edges, while ``linear`` uses grid points.
"""

from laddu.laddu import (
    LookupTable,
    LookupTableComplex,
    LookupTablePolar,
    LookupTableScalar,
)

__all__ = ['LookupTable', 'LookupTableComplex', 'LookupTablePolar', 'LookupTableScalar']
