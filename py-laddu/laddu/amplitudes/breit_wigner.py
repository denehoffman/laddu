"""Relativistic Breit-Wigner amplitude constructors.

Wraps the Rust implementation so it can be registered with an amplitude
:class:`laddu.amplitudes.Manager`.

Examples
--------
>>> from laddu.amplitudes import Manager
>>> from laddu.amplitudes.breit_wigner import BreitWigner
>>> manager = Manager()
>>> bw = manager.register(BreitWigner('rho', mass=0.775, width=0.149))  # doctest: +SKIP
"""

from laddu.laddu import BreitWigner

__all__ = ['BreitWigner']
