# Coupled-channel resonance fit

A coupled-channel analysis fits common resonance pole parameters to multiple
production or decay channels while allowing channel-specific couplings and
acceptances. The essential laddu mechanism is parameter sharing by name across
the models used by different likelihood terms.

Suppose two final states contain the same resonance. Build their kinematic
variables from their own channels and datasets, then construct line shapes with
identically declared pole parameters:

```python
import laddu as ld

pole_mass = ld.parameter(
    "pole_mass", initial=1.50, bounds=(1.30, 1.70), unit="GeV"
)
pole_width = ld.parameter(
    "pole_width", initial=0.15, bounds=(0.01, 0.50), unit="GeV"
)

line_a = resonance_line_shape(mass_a, pole_mass, pole_width)
line_b = resonance_line_shape(mass_b, pole_mass, pole_width)

coupling_a = ld.complex(
    ld.parameter("channel_a_re", fixed=1.0),
    ld.parameter("channel_a_im", fixed=0.0),
)
coupling_b = ld.polar_complex(
    ld.parameter("channel_b_mag", initial=0.5, bounds=(0.0, None)),
    ld.parameter(
        "channel_b_phase",
        initial=0.0,
        bounds=(-3.141592653589793, 3.141592653589793),
        periodic=True,
    ),
)

wave_a = (coupling_a * line_a * angular_a).tagged("shared_resonance_a")
wave_b = (coupling_b * line_b * angular_b).tagged("shared_resonance_b")
model_a = ld.Model(wave_a.norm_sqr())
model_b = ld.Model(wave_b.norm_sqr())
```

The fixed channel-A coupling defines the overall magnitude and phase
convention. `pole_mass` and `pole_width` are shared because both compiled
models register compatible parameters with those names; the channel couplings
remain independent.

Construct one likelihood term with each channel's own observed and accepted-MC
datasets:

```python
term_a = ld.NLL(model_a, data_a, accepted_mc_a, name="channel_a")
term_b = ld.NLL(model_b, data_b, accepted_mc_b, name="channel_b")
joint = ld.Likelihood([term_a, term_b], execution=execution)

fit = joint.fit(
    ld.ganesh.LBFGSBConfig(),
    initial={"pole_mass": 1.52, "pole_width": 0.12},
    terminators=[ld.ganesh.MaxSteps(1000)],
)
```

Use `ExtendedNLL` terms instead when the channel yields carry information.
Then include channel scales and normalize each accepted-MC sample to its
represented exposure, as described in {doc}`simultaneous-fits`.

## Diagnosing the coupling

Project and inspect each named likelihood term separately:

```python
projection_a = joint.projection(
    "channel_a", generated_mc_a, ["shared_resonance_a"]
)
projection_b = joint.projection(
    "channel_b", generated_mc_b, ["shared_resonance_b"]
)

weights_a = projection_a.weights(fit.x, acceptance_corrected=True)
weights_b = projection_b.weights(fit.x, acceptance_corrected=True)
```

Useful closure tests include fitting each channel alone, fitting both together,
releasing the shared pole parameters one at a time, and injecting deliberately
different pole values to verify that the joint fit detects tension. A shared
parameter name is a physical constraint, so incompatible definitions are
reported as an error rather than silently selecting one model's metadata.
