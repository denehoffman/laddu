# Particles, channels, and quantum numbers

Reaction topology connects the names in a dataset to their physical roles.
Quantum-number types keep integer and half-integer values exact while the
channel supplies invariant masses, frames, and decay angles.

## Exact angular momentum

```python
from fractions import Fraction

import laddu as ld

photon_spin = ld.S(1)
nucleon_spin = ld.S(Fraction(1, 2))
resonance_spin = ld.J(1.5)

projections = resonance_spin.projections()
allowed_totals = photon_spin.coupled_with(nucleon_spin)

assert resonance_spin.doubled == 3
assert resonance_spin.can_couple_to(photon_spin, nucleon_spin)
```

Use {py:class}`laddu.J` for total angular momentum,
{py:class}`laddu.S` for spin, {py:class}`laddu.L` for integral orbital
momentum, and {py:class}`laddu.M` for signed projections or helicities.
`L.parity` is $(-1)^L$.

```python
d_wave = ld.L(2)
assert d_wave.parity == ld.Parity.POSITIVE
```

## Particles

The built-in catalog contains common particles:

```python
photon = ld.particles.PHOTON
proton = ld.particles.PROTON
kaon = ld.particles.K_SHORT
```

Create an explicit particle when the catalog does not encode the state or
metadata required by the analysis:

```python
x_state = ld.Particle(
    "X",
    species="X",
    self_conjugate=True,
    spin=2,
    parity="+",
    statistics=ld.Statistics.BOSON,
)
```

Particle metadata is used by selection rules and generation thresholds; event
four-vectors still come from named dataset columns.

## Reaction channels

An edge describes an initial, intermediate, or final-state particle. A vertex
connects incoming and outgoing edges. This channel represents
$\gamma p\to Xp$, $X\to K_S^0K_S^0$:

```python
channel = ld.Channel(
    "gamma p -> K_S K_S p",
    edges=[
        ld.Edge("gamma", p4="gamma", particle=photon, output=True),
        ld.Edge("target", p4="target", particle=proton, output=True),
        ld.Edge("X", particle=x_state),
        ld.Edge("recoil", p4="recoil", particle=proton, output=True),
        ld.Edge("ks1", p4="ks1", particle=kaon, output=True),
        ld.Edge("ks2", p4="ks2", particle=kaon, output=True),
    ],
    vertices=[
        ld.Vertex(
            "production",
            incoming=["gamma", "target"],
            outgoing=["X", "recoil"],
        ),
        ld.Vertex("decay", incoming=["X"], outgoing=["ks1", "ks2"]),
    ],
)
```

The `p4` names must match the dataset schema. Intermediate four-vectors such as
`X` are reconstructed from the topology.

## Kinematic expressions and frames

Channel helpers return symbolic expressions; no event data are traversed yet.

```python
mass_x = channel.mass("X")
s_x = channel.s("X")

production = channel.vertex("production")
decay = channel.vertex("decay")

beam_axis = production.vec3("gamma")
helicity_axis = production.vec3("X")
production_normal = beam_axis.cross(helicity_axis)

theta_h = decay.theta("ks1", z_axis=helicity_axis, y_hint=production_normal)
phi_h = decay.phi("ks1", z_axis=helicity_axis, y_hint=production_normal)
```

Define axes once from a documented frame convention and reuse them in model
construction and validation. The next chapter applies explicit conservation
rules before amplitudes are built.
