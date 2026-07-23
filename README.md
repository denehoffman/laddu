<p align="center">
  <img src="docs/_static/logo.svg" alt="laddu" width="150">
</p>

# laddu

**Amplitude analysis made short and sweet.**

laddu is a Python library for constructing, evaluating, generating, and fitting multibody amplitude models. Its symbolic Python interface is backed by a high-performance Rust runtime with automatic differentiation, multithreaded CPU execution, JIT compilation, WGPU acceleration, and an optional MPI distribution.

> laddu is under active development. Pin a version for production analyses and record `laddu.capabilities()` with analysis outputs.

## Installation

laddu requires Python 3.11 or later. Install the standard build from PyPI:

```bash
python -m pip install laddu
```

For an MPI-enabled installation, use the optional extra in an environment with a working MPI implementation:

```bash
python -m pip install "laddu[mpi]"
```

To build the current checkout, install Rust, `uv`, and `just`, then build the extension:

```bash
just python-dev
```

The recipes create and use a project-local `.venv`. `nix develop` is optional; it
provides the same tools plus the native MPI, Vulkan, and GPU environment.

Confirm the installed backend and optional features before starting a large job:

```python
import laddu as ld

print(ld.backend())
print(ld.capabilities())
```

## A first model

laddu models are expression graphs. Parameters carry optimizer metadata, while event-dependent expressions are compiled once for the chosen runtime:

```python
import laddu as ld

s = ld.scalar("s")
mass = ld.parameter("mass", initial=1.5, bounds=(1.3, 1.7))
width = ld.parameter("width", initial=0.1, bounds=(0.01, 0.3))
amplitude = 1.0 / (mass**2 - s - 1j * mass * width)
model = ld.Model(amplitude.norm_sqr())

print(model.parameter_names)
```

Real analyses usually obtain invariants and angles from a `Channel`, then combine line shapes, angular functions, and complex production couplings. The following is the central fit pattern once `model`, observed `data`, and accepted normalization `mc` have been prepared:

```python
execution = ld.Execution("auto", precision="f64", autodiff="forward")
likelihood = ld.Likelihood(
    [ld.NLL(model, data, mc, name="signal")],
    execution=execution,
)
fit = likelihood.fit(
    ld.ganesh.LBFGSBConfig(history_size=10),
    initial=likelihood.sample_parameters(seed=7),
    terminators=[ld.ganesh.MaxSteps(500)],
)
result = dict(zip(fit.parameter_names, fit.x, strict=True))
```

The normalized unbinned objective is

$$
-\log \mathcal L(\boldsymbol\theta)
=-\sum_{i\in\mathrm{data}} w_i\log I(\Omega_i;\boldsymbol\theta)
+\left(\sum_i w_i\right)
\log\!\left[
\frac{\sum_{j\in\mathrm{accepted\ MC}} w_j I(\Omega_j;\boldsymbol\theta)}
{\sum_j w_j}
\right].
$$

## Data and Monte Carlo

Datasets can be constructed from NumPy arrays or read lazily from Parquet and ROOT. Four-vectors use `(E, px, py, pz)` order.

```python
import numpy as np

events = ld.Dataset.from_arrays(
    p4s={"beam": np.array([[9.0, 0.0, 0.0, 9.0]])},
    scalars={"run": np.array([12001.0])},
)
events.write_to(ld.ParquetSink("events.parquet"))
events = ld.read_parquet("events.parquet", chunk_size=100_000)
```

Given a channel with generation proposals, phase-space and modeled samples use the same generator:

```python
generator = ld.Generator(channel)
normalization_mc, report = generator.weighted(100_000, seed=10)
pseudo_data, report = generator.unweighted(
    10_000,
    model,
    parameters={"magnitude": 0.7, "phase": 1.2},
    seed=11,
    grow_envelope=True,
)
```

## Execution choices

Use `Execution("cpu")` for predictable local work, `Execution("jit")` for compiled CPU kernels, or `Execution("gpu")` for WGPU. `Execution("auto")` selects a sensible local CPU strategy. The MPI distribution supplies the same `laddu` module and enables distributed execution under `mpiexec`; dataset partitioning may be contiguous, file-group based, or row based.

## Documentation and development

The documentation includes task-oriented tutorials for I/O, Monte Carlo generation, fitting, mass-independent binned fitting, polarized photoproduction, cross sections, MPI, and runtime tuning. Build it locally with:

```bash
just docs-install
just docs-build
just docs-serve       # http://127.0.0.1:8000
```

Build the Rust API documentation with the KaTeX header, or build and open it in the default browser:

```bash
just rust-docs-build
just rust-docs-open
```

Useful development checks are:

```bash
just check-python-rust
just test-rust
just example-quick cpu
```

laddu is dual-licensed under MIT or Apache-2.0. Bug reports and focused pull requests are welcome through the GitHub repository.
