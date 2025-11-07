"""Bindings for variable extractors (invariant masses, Mandelstam variables, etc.).

These helpers wrap the lower-level Rust selectors and allow Python analyses to
bind derived quantities by name. For example, the mass of a two-kaon system can
be constructed with ``Mass(['kshort1', 'kshort2'])`` and then registered inside
an :class:`laddu.extensions.NLL`.

Examples
--------
>>> from laddu import Dataset
>>> from laddu.utils.variables import Mass
>>> columns = {
...     'kshort1_px': [0.1], 'kshort1_py': [0.0], 'kshort1_pz': [0.2], 'kshort1_e': [0.3],
...     'kshort2_px': [-0.1], 'kshort2_py': [0.0], 'kshort2_pz': [0.1], 'kshort2_e': [0.25],
... }
>>> dataset = Dataset.from_dict(columns)
>>> mass = Mass(['kshort1', 'kshort2'])
>>> mass  # doctest: +ELLIPSIS
Mass(...)
"""

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
