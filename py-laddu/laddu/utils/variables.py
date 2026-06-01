"""Bindings for variable extractors (invariant masses, Mandelstam variables, etc.).

These helpers wrap the lower-level Rust selectors and allow Python analyses to
bind derived quantities by name. Topology-dependent variables are constructed
from a :class:`laddu.Channel`.

Examples
--------
>>> import laddu as ld
>>> columns = {
...     'kshort1_px': [0.1], 'kshort1_py': [0.0], 'kshort1_pz': [0.2], 'kshort1_e': [0.3],
...     'kshort2_px': [-0.1], 'kshort2_py': [0.0], 'kshort2_pz': [0.1], 'kshort2_e': [0.25],
... }
>>> dataset = ld.io.from_dict(columns)
>>> channel = ld.Channel()
>>> _ = channel.create_decay('kk_decay', 'kk', ['kshort1', 'kshort2'])
>>> mass = channel.mass('kk')
>>> isinstance(mass, ld.Mass)
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laddu.laddu import (
    Angles,
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    Polarization,
    PolMagnitude,
    VariableExpression,
)

if TYPE_CHECKING:
    from laddu.amplitude import Expression

    _ScalarVariable = Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam


def _as_expression(self: _ScalarVariable, *tags: str) -> Expression:
    """Convert this variable into a real-valued expression."""
    from laddu.amplitudes.scalar import VariableScalar

    return VariableScalar(*tags, variable=self)


_AS_EXPRESSION_NAME = 'as_expression'

for _VariableType in (
    Mass,
    CosTheta,
    Phi,
    PolAngle,
    PolMagnitude,
    Mandelstam,
):
    setattr(_VariableType, _AS_EXPRESSION_NAME, _as_expression)

del _AS_EXPRESSION_NAME, _VariableType


__all__ = [
    'Angles',
    'CosTheta',
    'Mandelstam',
    'Mass',
    'Phi',
    'PolAngle',
    'PolMagnitude',
    'Polarization',
    'VariableExpression',
]
