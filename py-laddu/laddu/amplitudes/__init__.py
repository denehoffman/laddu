"""High-level amplitude construction helpers.

This module re-exports the Rust-backed amplitude building blocks as a cohesive Python API.

Examples
--------
>>> from laddu.amplitudes import Manager, amplitude_sum, common
>>> manager = Manager()
>>> scalar = manager.register(common.Scalar('mag'))  # overall magnitude
>>> ρ = manager.register(common.ComplexScalar('rho'))
>>> model = manager.model(amplitude_sum([scalar * ρ]))

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
