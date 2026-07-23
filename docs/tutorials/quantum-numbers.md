# Quantum numbers and angular momentum

laddu represents integer and half-integer quantum numbers exactly. Constructors
accept integers, exact half-integer floats, and {py:class}`fractions.Fraction`;
the `doubled` property is useful when an exact integer representation is needed.

```python
from fractions import Fraction

import laddu as ld

photon = ld.S(1)
nucleon = ld.S(Fraction(1, 2))
resonance = ld.J(1.5)

assert resonance.doubled == 3
assert resonance.multiplicity == 4
assert resonance.projections() == [
    ld.M(-1.5), ld.M(-0.5), ld.M(0.5), ld.M(1.5)
]
```

## Coupling angular momenta

Use `coupled_with` to enumerate triangle-rule results and `can_couple_to` to
test a proposed total directly:

```python
totals = photon.coupled_with(nucleon)
# [J(1/2), J(3/2)]

assert ld.J(1.5).can_couple_to(photon, nucleon)
assert not ld.J(0).can_couple_to(photon, nucleon)
```

Both {py:class}`laddu.J` and {py:class}`laddu.S` provide these methods. The
returned totals are `J` values, which makes them suitable for a production or
decay vertex without a separate top-level helper.

## Orbital momentum, parity, and projections

{py:class}`laddu.L` is always integral. Its string representation uses
spectroscopic notation, and its `parity` property returns $(-1)^L$.

```python
d_wave = ld.L(2)
assert str(d_wave) == "D"
assert d_wave.parity == ld.Parity.POSITIVE

for projection in d_wave.projections():
    print(projection)
```

Use {py:class}`laddu.M` for signed helicities and magnetic projections.
Subtraction and negation remain exact, which avoids floating-point equality
checks in Clebsch–Gordan and Wigner-function loops.

## Angular functions

The angular helpers accept the typed quantum numbers directly:

```python
coefficient = ld.clebsch_gordan(
    photon,
    ld.M(1),
    nucleon,
    ld.M(-0.5),
    ld.J(1.5),
    ld.M(0.5),
)

rotation = ld.WignerD(ld.J(1), ld.M(1), ld.M(0))
angular_factor = rotation.D(phi, theta, 0.0)
```

Typed values make invalid integral/half-integral combinations fail near model
construction instead of surfacing later during event evaluation.
