# ruff: noqa: RUF002
"""High-level amplitude construction helpers.

This module re-exports the Rust-backed amplitude building blocks as a cohesive Python API.

Examples
--------
>>> from laddu.amplitudes import parameter, scalar
>>> scalar_amp = scalar.Scalar('mag', parameter('mag'))  # overall magnitude
>>> rho = scalar.ComplexScalar('rho', (parameter('rho_re'), parameter('rho_im')))
>>> expr = scalar_amp * rho
>>> expr
×
├─ mag(id=0)
└─ rho(id=1)
<BLANKLINE>

Use :mod:`laddu.amplitudes.resonance` or the other submodules for concrete physics models.
"""

from laddu.amplitude import (
    CompiledExpression,
    Evaluator,
    Expression,
    One,
    Parameter,
    TestAmplitude,
    Zero,
    expr_product,
    expr_sum,
    parameter,
)

from . import (
    angular,
    kmatrix,
    lookup,
    resonance,
    scalar,
)

__all__ = [
    'CompiledExpression',
    'Evaluator',
    'Expression',
    'One',
    'Parameter',
    'TestAmplitude',
    'Zero',
    'angular',
    'expr_product',
    'expr_sum',
    'kmatrix',
    'lookup',
    'parameter',
    'resonance',
    'scalar',
]
