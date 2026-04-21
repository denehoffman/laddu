"""The Flatté amplitude constructor.

This helper returns ``laddu.Expression`` objects that can be
loaded directly and evaluated.

Examples
--------
>>> from laddu.amplitudes.flatte import Flatte
>>> from laddu import Mass, parameter
>>> expr = Flatte(
...     'a0_980',
...     mass=parameter('a0_980_mass', 0.980),
...     observed_channel_coupling=parameter('g_obs', 0.7),
...     alternate_channel_coupling=parameter('g_alt'),
...     observed_channel_daughter_masses=(Mass(["p1"]), Mass(["p2"])),
...     alternate_channel_daughter_masses=(0.1349768, 0.547862),
...     resonance_mass=Mass(["p1", "p2"]),
... )
>>> expr.norm_sqr()
NormSqr
└─ a0_980(id=0)
<BLANKLINE>
"""

from laddu.laddu import Flatte

__all__ = ['Flatte']
