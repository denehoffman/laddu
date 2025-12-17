# ruff: noqa: RUF002
"""Elementary scalar amplitude components.

``Scalar`` returns a real-valued scaling parameter, ``ComplexScalar`` exposes
independent real and imaginary parameters, and ``PolarComplexScalar`` uses a
magnitude/phase parameterisation. They are typically combined with dynamical
amplitudes like :mod:`laddu.amplitudes.breit_wigner`.

Examples
--------
>>> from laddu.amplitudes import common, parameter
>>> mag = common.Scalar('mag', parameter('mag'))
>>> phase = common.PolarComplexScalar('cplx', parameter('r'), parameter('theta'))
>>> mag * phase
×
├─ mag(id=0)
└─ cplx(id=1)
<BLANKLINE>
"""

from laddu.laddu import ComplexScalar, PolarComplexScalar, Scalar

__all__ = ['ComplexScalar', 'PolarComplexScalar', 'Scalar']
