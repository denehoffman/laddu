# Sharing resonance parameters across channels

A coupled-channel fit is a simultaneous fit whose datasets describe different
reaction or decay channels. Pole parameters are shared, while kinematics,
acceptance, and production couplings remain channel-specific.

## Build channel-specific line shapes

Use identically named parameters in both models:

```python
pole_mass = ld.parameter(
    "pole_mass",
    initial=1.50,
    bounds=(1.30, 1.70),
    unit="GeV",
)
pole_width = ld.parameter(
    "pole_width",
    initial=0.15,
    bounds=(0.01, 0.50),
    unit="GeV",
)

daughter_a1 = channel_a.particle("a1")
daughter_a2 = channel_a.particle("a2")
daughter_b1 = channel_b.particle("b1")
daughter_b2 = channel_b.particle("b2")

line_a = ld.relativistic_breit_wigner(
    channel_a.s("X"),
    mass=pole_mass,
    width=pole_width,
    mass1=daughter_a1.mass,
    mass2=daughter_a2.mass,
    l=0,
)
line_b = ld.relativistic_breit_wigner(
    channel_b.s("X"),
    mass=pole_mass,
    width=pole_width,
    mass1=daughter_b1.mass,
    mass2=daughter_b2.mass,
    l=0,
)
```

Each channel gets its own complex production coupling:

```python
coupling_a = ld.complex(
    ld.parameter("channel_a_re", fixed=1.0),
    ld.parameter("channel_a_im", fixed=0.0),
)
coupling_b = ld.polar_complex(
    ld.parameter("channel_b_magnitude", initial=0.5, bounds=(0.0, None)),
    ld.parameter(
        "channel_b_phase",
        initial=0.0,
        bounds=(-3.141592653589793, 3.141592653589793),
        periodic=True,
    ),
)

model_a = ld.Model((coupling_a * line_a).norm_sqr())
model_b = ld.Model((coupling_b * line_b).norm_sqr())
```

The fixed channel-A coupling defines the overall magnitude and phase
convention. `pole_mass` and `pole_width` are shared by name; channel couplings
are independent.

## Compose and fit the channel likelihoods

```python
joint = ld.Likelihood(
    [
        ld.NLL(model_a, data=data_a, accepted_mc=accepted_mc_a, name="channel_a"),
        ld.NLL(model_b, data=data_b, accepted_mc=accepted_mc_b, name="channel_b"),
    ]
)

fit = joint.fit(
    initial={"pole_mass": 1.52, "pole_width": 0.12},
    terminators=[ld.ganesh.MaxSteps(1000)],
)
```

Use `ExtendedNLL` when relative or absolute channel yields carry information;
then normalize each accepted-MC sample to its represented exposure as described
in {doc}`likelihoods` and {doc}`simultaneous-fits`.

Diagnose the coupling by fitting channels separately and jointly, releasing
shared parameters one at a time, and injecting incompatible pole values in
pseudo-data. Parameter sharing is a physical constraint, not merely a software
convenience.
