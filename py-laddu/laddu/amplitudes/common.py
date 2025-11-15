# ruff: noqa: RUF002
"""Elementary scalar amplitude components.

``Scalar`` returns a real-valued scaling parameter, ``ComplexScalar`` exposes
independent real and imaginary parameters, and ``PolarComplexScalar`` uses a
magnitude/phase parameterisation. They are typically combined with dynamical
amplitudes in :mod:`laddu.amplitudes.breit_wigner`.

Examples
--------
>>> from laddu.amplitudes import Manager, amplitude_sum, common, parameter
>>> manager = Manager()
>>> mag = manager.register(common.Scalar('mag', parameter('mag')))
>>> phase = manager.register(
...     common.PolarComplexScalar('cplx', parameter('r'), parameter('theta'))
... )
>>> amplitude_sum([mag * phase])
×
├─ mag(id=0)
└─ cplx(id=1)
<BLANKLINE>
"""

from laddu.laddu import ComplexScalar, PolarComplexScalar, Scalar

__all__ = ['ComplexScalar', 'PolarComplexScalar', 'Scalar']
