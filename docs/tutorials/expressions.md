# Expressions, amplitudes, and models

laddu expressions are immutable symbolic graphs. Arithmetic constructs the
graph; numerical work starts when an expression is evaluated on a dataset or
compiled into a {py:class}`laddu.Model`.

## Event values and parameters

```python
mass = generation_channel.mass("X")
s = generation_channel.s("X")

mass_0 = ld.parameter(
    "mass_0", initial=1.50, bounds=(1.35, 1.65), unit="GeV"
)
width_0 = ld.parameter(
    "width_0", initial=0.12, bounds=(0.01, 0.30), unit="GeV"
)
```

Event expressions vary by row. Parameters vary during fitting. Constants are
lifted into expressions automatically.

## Vectors, frames, and angular functions

Channel and vertex helpers preserve the topology used to define a frame:

```python
production = generation_channel.vertex("production")
decay = generation_channel.vertex("decay")

beam_axis = production.vec3("gamma")
helicity_axis = production.vec3("X")
normal = beam_axis.cross(helicity_axis)

theta = decay.theta("ks1", z_axis=helicity_axis, y_hint=normal)
phi = decay.phi("ks1", z_axis=helicity_axis, y_hint=normal)

angular = ld.WignerD(ld.J(2), ld.M(0), ld.M(0)).D(
    alpha=phi,
    beta=theta,
).conj()
```

{py:class}`laddu.Vec3` uses Euclidean operations. {py:class}`laddu.Vec4` uses
the $(+---)$ metric and positional order $(E,p_x,p_y,p_z)$:

```python
parent = ld.Vec4.event("ks1") + ld.Vec4.event("ks2")
daughter_in_parent_frame = ld.Vec4.event("ks1").boost(-parent.beta())
```

## Line shapes and coherent amplitudes

Built-in amplitude functions return complex expressions. For an $S$-wave
two-body resonance,

```python
m_k = ld.particles.K_SHORT.mass
line_shape = ld.relativistic_breit_wigner(
    s,
    mass=mass_0,
    width=width_0,
    mass1=m_k,
    mass2=m_k,
    l=0,
)

second_magnitude = ld.parameter(
    "second_magnitude", initial=0.3, bounds=(0.0, 3.0)
)
second_phase = ld.parameter(
    "second_phase",
    initial=0.0,
    bounds=(-3.141592653589793, 3.141592653589793),
    periodic=True,
)

reference_wave = line_shape.tagged("reference")
second_wave = (
    ld.polar_complex(second_magnitude, second_phase) * angular
).tagged("second")

amplitude = reference_wave + second_wave
intensity = amplitude.norm_sqr()
model = ld.Model(intensity)
```

Amplitudes leading to the same observed quantum state add coherently before
`norm_sqr`. Orthogonal unobserved states contribute separate intensities that
are added incoherently.

Tags are structural labels. A projection can retain selected amplitudes while
preserving their mutual interference:

```python
selected_intensity = intensity.project(["reference", "second"])
```

## Inspect a model before fitting

```python
parameters = model.default_parameters
values = model.evaluate(generated_mc, parameters=parameters, real=True)
```

Check that the intensity is finite and positive over a large generated sample,
that parameter names and bounds match the intended convention, and that tagged
projections behave as expected. The next chapter uses this model as a target
density for pseudo-data generation.
