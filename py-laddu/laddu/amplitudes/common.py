"""Elementary scalar amplitude components.

``Scalar`` returns a real-valued scaling parameter, ``ComplexScalar`` exposes
independent real and imaginary parameters, and ``PolarComplexScalar`` uses a
magnitude/phase parameterisation. They are typically combined with dynamical
amplitudes in :mod:`laddu.amplitudes.breit_wigner`.

Examples
--------
>>> from laddu.amplitudes import Manager, amplitude_sum
>>> from laddu.amplitudes import common
>>> manager = Manager()
>>> mag = manager.register(common.Scalar('mag'))
>>> phase = manager.register(common.PolarComplexScalar('cplx'))
>>> expr = amplitude_sum([mag * phase])
>>> model = manager.model(expr)
"""

from laddu.laddu import ComplexScalar, PolarComplexScalar, Scalar

__all__ = ['ComplexScalar', 'PolarComplexScalar', 'Scalar']
