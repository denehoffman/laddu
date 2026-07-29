# Fitting an unbinned model

This chapter assumes that `model`, observed `data`, and accepted normalization `mc` share a compatible schema. The normalization sample must pass the same reconstruction and selection as the data.

## Build the objective

For a normalized intensity, laddu minimizes

$$
\mathcal F(\boldsymbol\theta)
=-\sum_{i=1}^{N_\mathrm{data}} w_i\log I(\Omega_i;\boldsymbol\theta)
+\left(\sum_i w_i\right)\log \widehat{\mathcal N}(\boldsymbol\theta),
$$

where $\widehat{\mathcal N}$ is a weighted accepted-MC integral. Overall intensity scale cancels, so fix one complex coupling as the reference amplitude.

```python
import laddu as ld
import numpy as np

execution = ld.Execution("auto", precision="f64", autodiff="forward")
term = ld.NLL(model, data, mc, name="signal")
likelihood = ld.Likelihood([term], execution=execution)

initial = likelihood.sample_parameters(seed=100)
value, gradient = likelihood.value_and_gradient(initial)
assert np.isfinite(value) and np.all(np.isfinite(gradient))
```

## Fixing and freeing parameters

A parameter can be fixed when it is created:

```python
reference_re = ld.parameter("reference_re", fixed=1.0)
reference_im = ld.parameter("reference_im", fixed=0.0)
```

Fixed parameters remain part of the expression but are absent from
`model.parameter_names` and optimizer vectors. This is the usual way to remove
the unobservable overall magnitude and phase of a coherent amplitude.

Models also support immutable post-construction changes:

```python
nominal = ld.Model(intensity)
mass_fixed = nominal.fix("resonance_mass", 1.50)
mass_freed = mass_fixed.free("resonance_mass")

assert "resonance_mass" not in mass_fixed.parameter_names
assert "resonance_mass" in mass_freed.parameter_names
```

`fix` and `free` return recompiled models; they do not mutate the original.
Build a new likelihood from the returned model because the parameter layout and
gradient dimension have changed. Bounds and metadata are retained when a
parameter is freed again.

Use fixing for an intentional physical constraint or a staged fit, not to hide
an unstable direction. Record the fixed values alongside the fit result.

## Minimize

L-BFGS-B uses model bounds and analytic automatic derivatives:

```python
fit = likelihood.fit(
    ld.ganesh.LBFGSBConfig(history_size=10),
    initial=initial,
    terminators=[ld.ganesh.MaxSteps(500)],
    observers=[ld.ganesh.ProgressObserver(interval=10)],
)
assert isinstance(fit.x, np.ndarray)
fitted = dict(zip(fit.parameter_names, fit.x, strict=True))
print(fit)
```

`initial` accepts a Python sequence, a one-dimensional NumPy array, or a
dictionary keyed by free-parameter name. A dictionary starts from parameter
defaults and overrides only the named entries:

```python
fit = likelihood.fit(
    ld.ganesh.LBFGSBConfig(),
    initial={"resonance_mass": 1.52, "resonance_width": 0.12},
)
```

Unknown dictionary keys are errors, which catches misspelled parameter names.
`ganesh.VectorInit` remains available when interacting with ganesh directly,
but is unnecessary for `Likelihood.fit`.

Nelder–Mead normally creates a scaled orthogonal simplex around `initial`.
Supply all $n+1$ vertices explicitly through its configuration when the
simplex geometry matters:

```python
center = np.asarray(initial, dtype=np.float64)
steps = 0.01 * np.maximum(np.abs(center), 1.0)
simplex = np.vstack([center, center + np.diag(steps)])

fit = likelihood.fit(
    ld.ganesh.NelderMeadConfig(initial_simplex=simplex),
    initial=center,
    terminators=[ld.ganesh.MaxSteps(2000)],
)
```

The simplex and `initial` are expressed in the same free-parameter order.

Run multiple reproducibly seeded starts. Agreement in objective value is more meaningful than agreement in periodic phases or parameters related by model symmetries.

## MCMC sampling

The affine-invariant ensemble sampler requires a two-dimensional matrix with
one free-parameter vector per walker. A small cloud around a converged minimum
is a useful initialization for a local posterior exploration:

```python
n_walkers = max(32, 2 * len(fit.x))
walkers = likelihood.walker_positions(
    fit.x,
    n_walkers,
    scale=1.0e-3,
    seed=2026,
)

samples = likelihood.sample(
    ld.ganesh.AIESConfig(),
    ld.ganesh.AIESInit(walkers),
    seed=2027,
    terminators=[ld.ganesh.MaxSteps(5000)],
)

chain = samples.chain  # NumPy array: (walkers, steps, parameters)
```

`walker_positions` uses each parameter's declared optimizer scale when one is
available, otherwise `max(abs(center), 1)`. It resamples bounded coordinates
and wraps periodic coordinates. This keeps the convenience API small: it
generates positions only, while ganesh remains responsible for validating and
running the ensemble. Use a broader or physics-informed ensemble when exploring
separated modes; a tiny cloud around one minimum will not discover them by
itself. Discard burn-in, check acceptance and autocorrelation diagnostics, and
run enough independent ensembles to assess convergence.

## Projection and uncertainty

Tag amplitude components before constructing `model`, then project them on generated MC:

```python
projection = likelihood.projection("signal", generated_mc, ["wave_0", "wave_2"])
weights = projection.weights(fitted, acceptance_corrected=True)
```

The projection preserves coherent interference among the retained tags. A sum of single-wave projections is generally not the coherent total.

For a robust first uncertainty estimate, bootstrap the data and refit from the central solution:

```python
replicas = []
for seed in range(1000, 1100):
    replica_data = data.bootstrap(seed=seed)
    replica_likelihood = ld.Likelihood(
        [ld.NLL(model, replica_data, mc, name="signal")], execution=execution
    )
    result = replica_likelihood.fit(
        ld.ganesh.LBFGSBConfig(history_size=10),
        initial=fitted,
        terminators=[ld.ganesh.MaxSteps(300)],
    )
    replicas.append(result.x)
```

Inspect convergence state, gradients, boundary contacts, start-to-start stability, projection agreement, and pull behavior on closure samples before interpreting physical parameters.
