# ruff: noqa: RUF002
"""Elementary scalar amplitude components.

``Scalar`` returns a real-valued scaling parameter, ``VariableScalar`` evaluates
an event variable, ``ComplexScalar`` exposes independent real and imaginary
parameters, and ``PolarComplexScalar`` uses a magnitude/phase parameterisation.
They are typically combined with dynamical amplitudes like
:mod:`laddu.amplitudes.resonance`.

Examples
--------
>>> from laddu.amplitudes import parameter, scalar
>>> mag = scalar.Scalar('mag', parameter('mag'))  # or just scalar.Scalar('mag')
>>> phase = scalar.PolarComplexScalar('cplx', (parameter('r'), parameter('theta')))
>>> mag * phase
×
├─ mag(id=0)
└─ cplx(id=1)
<BLANKLINE>
"""

from laddu.laddu import ComplexScalar, PolarComplexScalar, Scalar, VariableScalar

__all__ = ['ComplexScalar', 'PolarComplexScalar', 'Scalar', 'VariableScalar']
