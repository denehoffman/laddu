# Expressions, amplitudes, and models

laddu expressions are immutable symbolic graphs. Arithmetic constructs the
graph; a {py:class}`laddu.Model` evaluates it at parameter values, with a dataset
when the expression needs event inputs.

## Check values and gradients

Use {py:meth}`laddu.Model.evaluate` for a value and
{py:meth}`laddu.Model.value_and_gradient` for the value and its derivatives
with respect to the real free parameters. Omit the dataset (or pass `None`)
when the expression depends only on parameters and constants. These calls
return a Python scalar; no example event or fit is needed.

The following complete example checks a complex polynomial against known
values and derivatives, selects a scalar component of a matrix solve, and
evaluates an event-dependent expression on a one-event dataset. It can also be
downloaded as {download}`scalar_evaluation.py <../../python/examples/scalar_evaluation.py>`.
From the repository root with the development environment installed, run:

```bash
.venv/bin/python python/examples/scalar_evaluation.py
```

```{literalinclude} ../../python/examples/scalar_evaluation.py
:language: python
:start-at: import laddu
```

### Parameters and result shapes

Omitting `parameters` uses `model.default_parameters`. A partial dictionary
overrides only the named free parameters; other free parameters keep their
defaults. Ordered lists, tuples, and NumPy arrays follow
`model.parameter_names`, which is also the gradient order. Fixed parameters
retain their fixed values and do not occupy gradient entries.

| Dataset argument | Value | Gradient |
| --- | --- | --- |
| Omitted or `None` | Python `complex` | NumPy array `(n_free_parameters,)` |
| Supplied, including one event | NumPy array `(n_events,)` | NumPy array `(n_events, n_free_parameters)` |

With `real=True`, the value is a Python `float` without a dataset. Arrays use
`float64` instead of `complex128`. Both values and derivatives are projected
onto their real components; this does not compute their magnitudes. With no
free parameters, gradients have shape `(0,)` without a dataset or
`(n_events, 0)` with one. A supplied one-event dataset always retains its event
dimension.

### Inspect and update parameter definitions

The complete parameter declarations are available as detached
`model.parameter_specs` and `model.fixed_parameters` snapshots, including
parameters folded away by graph optimization:

```python
mass_0 = ld.parameter("mass_0", initial=1.50, bounds=(1.35, 1.65), unit="GeV")
width_0 = ld.parameter("width_0", initial=0.12, bounds=(0.01, 0.30), unit="GeV")
model = ld.Model(mass_0 + width_0)

spec = model.parameter_specs["mass_0"]
print(spec.initial, spec.bounds, spec.unit, spec.description)
print(model.fixed_parameters)
```

Each `ParameterSpec` record is immutable, and changing either returned mapping
does not change the model. Its fields are `name`, `fixed`, `initial`, `bounds`,
`periodic`, `scale`, `unit`, `latex`, and `description`. `initial` is a scalar,
an inclusive `(minimum, maximum)` initialization range, or `None` for the
default zero. `bounds` is an inclusive pair whose endpoints may be `None`, or
`None` when the parameter is unbounded.

To change parameter definitions, pass a batch of `ParameterUpdate` objects to
`with_parameters`. The method returns a new recompiled model and validates the
whole batch before applying any change. Updates affect every occurrence of a
name in the source graph, including occurrences removed by optimization:

```python
configured = model.with_parameters({
    "mass_0": ld.ParameterUpdate(
        fixed=1.50,
        bounds=(1.35, 1.65),
        unit="GeV",
    ),
    "width_0": ld.ParameterUpdate(initial=0.12, scale=0.1),
})

freed = configured.with_parameters({
    "mass_0": ld.ParameterUpdate(fixed=None),
})
```

Omitted update fields preserve their current values. Explicit `None` clears
optional metadata such as `unit`, `latex`, `description`, and `scale`; it
removes bounds when used for `bounds`, resets initialization to the default
zero when used for `initial`, and frees a parameter when used for `fixed`.
Fixing also sets the initial value to the fixed value unless `initial` is
explicitly supplied in the same update. Changing `initial` alone leaves the
fixed/free state unchanged. `periodic` is always a Boolean.
Unknown parameter names and batches that leave
an initial or fixed value outside its bounds raise `laddu.LadduError`, and the
original model remains unchanged.

### Scalar results and event inputs

Only scalar results are supported. Expressions can contain vectors, matrices,
and linear solves internally: select a vector component with `vector[i]` or a
matrix element with `matrix.at(i, j)` before constructing the model. A vector
or matrix root raises `laddu.LadduError`; whole-array values and Jacobians are
not returned by these methods.

An expression that reads a scalar column such as `ld.scalar("mass")` or a
four-momentum column needs a dataset containing that input. Calling either
model method without a dataset raises `laddu.LadduError`, identifying the
required input and asking for a dataset. Inputs are never filled with invented
values. Both positional `model.evaluate(dataset, ...)` and keyword
`model.evaluate(dataset=dataset, ...)` calls remain supported, as do the
corresponding `value_and_gradient` calls.

### Execution settings

Calls without a dataset use automatic CPU/JIT selection by default. Pass an
existing {py:class}`laddu.Execution` to choose the CPU interpreter or JIT and
supported precision and differentiation settings, for example:

```python
execution = ld.Execution("cpu", precision="f64", autodiff="forward")
value, gradient = model.value_and_gradient(execution=execution)
```

`autodiff` accepts `"auto"`, `"forward"`, or `"reverse"`; `precision` accepts
`"f32"` or `"f64"`. These control computation, while the result types remain
those in the table above. Explicit `"jit"` execution requires a build with JIT
support; inspect `ld.capabilities()["jit"]` when deciding whether to run a
JIT-specific check. Use `f64` and explicit numerical tolerances as a validation
baseline.

Explicit GPU execution is unsupported without a dataset and raises
`laddu.LadduError` instead of falling back to CPU. Constructing a GPU
`Execution` can itself fail when no adapter is available. Dataset execution
keeps its existing backend behavior. See {doc}`execution` for more on execution
configuration.

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
