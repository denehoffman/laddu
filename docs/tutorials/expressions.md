# Expressions, vectors, and linear algebra

laddu models are immutable expression graphs. Python operators build the graph;
numerical work begins only when an expression is evaluated against a dataset or
compiled into a {py:class}`laddu.Model`.

## Scalars, vectors, and matrices

Create event scalars by name and combine them with fit parameters or constants:

```python
import laddu as ld

x = ld.scalar("x")
slope = ld.parameter("slope", initial=1.0, bounds=(-5.0, 5.0))
intercept = ld.parameter("intercept", initial=0.0)
line = slope * x + intercept
```

{py:func}`laddu.vector` and {py:func}`laddu.matrix` accept expression-like
elements. Index a vector with `v[i]` and a matrix with `m.at(row, column)`.
The `shape` property reports `()`, `(n,)`, or `(rows, columns)`.

```python
v = ld.vector([x, 1.0])
a = ld.matrix([[slope, 0.0], [0.0, intercept]])
b = ld.matrix([[1.0, x], [x, 1.0]])

matrix_product = a @ b
matrix_vector_product = a @ v
scalar_product = ld.dot(v, v)
solution = ld.solve(a, v)
```

The `@` operator covers matrix–matrix multiplication, matrix–vector
multiplication, and vector dot products. `ld.dot(v, v)` remains available when
the named form reads more clearly, and {py:func}`laddu.solve` solves a linear
system. Shape mismatches are errors rather than implicit broadcasting.

## Symbolic three-vectors

{py:class}`laddu.Vec3` provides Euclidean vector algebra while keeping every
component symbolic:

```python
momentum = ld.Vec3.event("track")
beam_axis = ld.Vec3.z_axis()
normal = beam_axis.cross(momentum).unit()

p_parallel = momentum @ beam_axis  # equivalent to momentum.dot(beam_axis)
azimuth = momentum.phi()
p4_from_mass = momentum.with_mass(0.13957)
```

`Vec3.event("track")` reads the spatial components of the named event
four-vector. Channel and vertex helpers are usually preferable when a vector
already belongs to a reaction edge because they preserve the topology in model
code.

## Four-vectors and boosts

{py:class}`laddu.Vec4` uses the $(+---)$ metric. Constructors and positional
array rows consistently use metric order `(E, px, py, pz)`. Dataset columns are
named and therefore have no intrinsic order; this convention matters only when
one four-vector is represented by a positional four-element value, such as a
row passed to `Dataset.from_arrays`.

```python
parent = ld.Vec4.event("parent")
daughter = ld.Vec4.event("daughter")

mass = (parent - daughter).mass()
beta = parent.beta()
daughter_in_parent_frame = daughter.boost(-beta)
direction = daughter_in_parent_frame.momentum().unit()
lorentz_product = parent @ daughter  # equivalent to parent.dot(daughter)
```

Use `m2()` when a signed invariant is meaningful and `mass()` for the
nonnegative invariant mass. `mag()` and `mass()` are aliases, as are `mag2()`
and `m2()`; all four use the Lorentz invariant rather than a Euclidean
component norm.

## Complex expressions and projections

Complex couplings can be written in Cartesian or polar form:

```python
magnitude = ld.parameter("magnitude", initial=1.0, bounds=(0.0, None))
phase = ld.parameter("phase", initial=0.0, bounds=(-3.14159, 3.14159), periodic=True)
coupling_a = ld.polar_complex(magnitude, phase)
coupling_b = ld.complex(
    ld.parameter("wave_b_re", initial=0.2),
    ld.parameter("wave_b_im", initial=0.0),
)

wave_a = (coupling_a * amplitude_a).tagged("wave_a")
wave_b = (coupling_b * amplitude_b).tagged("wave_b")
intensity = (wave_a + wave_b).norm_sqr()
wave_a_only = intensity.project(["wave_a"])
```

Tags are structural metadata. They do not alter evaluation until a projection
selects them. Here `wave_a_only` removes `wave_b`; projecting both tags retains
their coherent interference.
