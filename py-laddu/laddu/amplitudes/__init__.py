# ruff: noqa: RUF002
"""High-level amplitude construction helpers.

This module re-exports the Rust-backed amplitude building blocks as a cohesive Python API.

Examples
--------
>>> from laddu.amplitudes import common, parameter
>>> scalar = common.Scalar('mag', parameter('mag'))  # overall magnitude
>>> rho = common.ComplexScalar('rho', parameter('rho_re'), parameter('rho_im'))
>>> expr = scalar * rho
>>> expr
×
├─ mag(id=0)
└─ rho(id=1)
<BLANKLINE>

Use :mod:`laddu.amplitudes.breit_wigner` or the other submodules for concrete physics models.
"""

from laddu.laddu import (
    Evaluator,
    Expression,
    One,
    ParameterLike,
    TestAmplitude,
    Zero,
    constant,
    expr_product,
    expr_sum,
    parameter,
)

from . import (
    breit_wigner,
    common,
    kmatrix,
    phase_space,
    piecewise,
    ylm,
    zlm,
)

__all__ = [
    'Evaluator',
    'Expression',
    'One',
    'ParameterLike',
    'TestAmplitude',
    'Zero',
    'breit_wigner',
    'common',
    'constant',
    'expr_product',
    'expr_sum',
    'kmatrix',
    'parameter',
    'phase_space',
    'piecewise',
    'ylm',
    'zlm',
]
