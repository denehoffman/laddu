"""High-level amplitude construction helpers.

This module re-exports the Rust-backed amplitude building blocks as a cohesive Python API.

Examples
--------
>>> from laddu.amplitudes import Manager, amplitude_sum, common, parameter
>>> manager = Manager()
>>> scalar = manager.register(common.Scalar('mag', parameter('mag')))  # overall magnitude
>>> rho = manager.register(
...     common.ComplexScalar('rho', parameter('rho_re'), parameter('rho_im'))
... )
>>> expr = amplitude_sum([scalar * rho])
>>> expr
×
├─ mag(id=0)
└─ rho(id=1)
<BLANKLINE>
>>> model = manager.model(expr)
>>> model  # doctest: +ELLIPSIS
<laddu.Model object at ...>

Use :mod:`laddu.amplitudes.breit_wigner` or the other submodules for concrete physics models.
"""

from laddu.amplitudes import (
    breit_wigner,
    common,
    kmatrix,
    phase_space,
    piecewise,
    ylm,
    zlm,
)
from laddu.laddu import (
    Amplitude,
    AmplitudeID,
    AmplitudeOne,
    AmplitudeZero,
    Evaluator,
    Expression,
    Manager,
    Model,
    ParameterLike,
    TestAmplitude,
    amplitude_product,
    amplitude_sum,
    constant,
    parameter,
)

__all__ = [
    'Amplitude',
    'AmplitudeID',
    'AmplitudeOne',
    'AmplitudeZero',
    'Evaluator',
    'Expression',
    'Manager',
    'Model',
    'ParameterLike',
    'TestAmplitude',
    'amplitude_product',
    'amplitude_sum',
    'breit_wigner',
    'common',
    'constant',
    'kmatrix',
    'parameter',
    'phase_space',
    'piecewise',
    'ylm',
    'zlm',
]
