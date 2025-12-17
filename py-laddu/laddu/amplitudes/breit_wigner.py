"""Relativistic Breit-Wigner amplitude constructors.

These helpers return ``laddu.Expression`` objects that can be
loaded directly and evaluated.

Examples
--------
>>> from laddu.amplitudes.breit_wigner import BreitWigner
>>> from laddu import Mass, constant
>>> expr = BreitWigner('rho', mass=constant('rho_mass', 0.775), width=constant('rho_width', 0.149), l=2, daughter_1_mass=Mass(["p1"]), daughter_2_mass=Mass(["p2"]), resonance_mass=Mass(["p1","p2"]))
>>> expr.norm_sqr()
NormSqr
└─ rho(id=0)
<BLANKLINE>
"""

from laddu.laddu import BreitWigner

__all__ = ['BreitWigner']
