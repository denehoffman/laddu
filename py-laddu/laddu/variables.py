"""Variable extractors for datasets and events."""

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


def _as_expression(self: _ScalarVariable, name: str) -> Expression:
    """Convert this variable into a real-valued expression."""
    from laddu.amplitudes.scalar import VariableScalar

    return VariableScalar(name, self)


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
