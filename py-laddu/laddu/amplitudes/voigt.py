"""The Voigt amplitude constructor.

This helper returns ``laddu.Expression`` objects that can be
loaded directly and evaluated.

Examples
--------
>>> from laddu.amplitudes.voigt import Voigt
>>> from laddu import Mass, parameter
>>> expr = Voigt(
...     'rho_voigt',
...     mass=parameter('rho_mass', 0.775),
...     width=parameter('rho_width', 0.149),
...     sigma=parameter('rho_sigma', 0.010),
...     resonance_mass=Mass(["p1", "p2"]),
... )
>>> expr.norm_sqr()
NormSqr
└─ rho_voigt(id=0)
<BLANKLINE>
"""

from laddu.laddu import Voigt

__all__ = ['Voigt']
