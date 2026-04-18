"""Bindings for variable extractors (invariant masses, Mandelstam variables, etc.).

These helpers wrap the lower-level Rust selectors and allow Python analyses to
bind derived quantities by name. For example, the mass of a two-kaon system can
be constructed with ``Mass(['kshort1', 'kshort2'])`` and then registered inside
an :class:`laddu.extensions.NLL`.

Examples
--------
>>> import laddu as ld
>>> from laddu.utils.variables import Mass
>>> columns = {
...     'kshort1_px': [0.1], 'kshort1_py': [0.0], 'kshort1_pz': [0.2], 'kshort1_e': [0.3],
...     'kshort2_px': [-0.1], 'kshort2_py': [0.0], 'kshort2_pz': [0.1], 'kshort2_e': [0.25],
... }
>>> dataset = ld.io.from_dict(columns)
>>> mass = Mass(['kshort1', 'kshort2'])
>>> mass
Mass { source: Selection(P4Selection { names: ["kshort1", "kshort2"], indices: [] }) }
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laddu.laddu import (
    Angles,
    CosTheta,
    Decay,
    Mandelstam,
    Mass,
    Particle,
    Phi,
    PolAngle,
    Polarization,
    PolMagnitude,
    Reaction,
    VariableExpression,
)

if TYPE_CHECKING:
    from laddu.amplitudes import Expression

    _ScalarVariable = Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam


def _as_expression(self: _ScalarVariable, name: str) -> Expression:
    """Convert this variable into a real-valued expression."""
    from laddu.amplitudes.common import VariableScalar

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
    'Decay',
    'Mandelstam',
    'Mass',
    'Particle',
    'Phi',
    'PolAngle',
    'PolMagnitude',
    'Polarization',
    'Reaction',
    'VariableExpression',
]
