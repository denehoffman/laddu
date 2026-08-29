# Fitting a model to unbinned event data

This chapter uses the `model` built in {doc}`expressions`, observed `data`, and
detector-selected `accepted_mc`. All three must have compatible schemas.

## The normalized event likelihood

For intensity $I(\Omega;\theta)$, laddu's normalized `NLL` minimizes

$$
\mathcal F(\theta)
=-\sum_{i\in\mathrm{data}}w_i\log I(\Omega_i;\theta)
+N_w\log \widehat{\mathcal N}(\theta),
\qquad
N_w=\sum_i w_i,
$$

where the accepted-MC estimate of the normalization is

$$
\widehat{\mathcal N}(\theta)
=\sum_{j\in\mathrm{accepted\ MC}}w_j I(\Omega_j;\theta).
$$

A global rescaling of $I$ cancels. Fix one complex amplitude's magnitude and
phase to define a convention, as the `reference_wave` did in the preceding
model.

## Build and inspect the objective

```python
term = ld.NLL(model, data=data, accepted_mc=accepted_mc, name="signal")
likelihood = ld.Likelihood([term])

initial = likelihood.sample_parameters(seed=100)
value, gradient = likelihood.value_and_gradient(initial)
```

Check that the initial objective and every gradient component are finite.
`likelihood.parameter_names` defines the order of all parameter vectors.

Parameters may instead be fixed in the expression or in a compiled model:

```python
reference_re = ld.parameter("reference_re", fixed=1.0)
reference_im = ld.parameter("reference_im", fixed=0.0)

mass_fixed_model = model.with_parameters({
    "mass_0": ld.ParameterUpdate(fixed=1.50),
})
mass_freed_model = mass_fixed_model.with_parameters({
    "mass_0": ld.ParameterUpdate(fixed=None),
})
```

`with_parameters` returns a new model. Rebuild the likelihood because its
parameter layout has changed. Several independent updates can be applied in
one call, and the entire batch is validated atomically.

## Minimize the likelihood

With no optimizer configuration, `fit` uses L-BFGS-B with its default
settings:

```python
fit = likelihood.fit(
    initial=initial,
    terminators=[ld.ganesh.MaxSteps(500)],
)

fitted = dict(zip(fit.parameter_names, fit.x, strict=True))
```

`initial` may be a Python sequence, a one-dimensional NumPy array of either
floating dtype, or a partial mapping by parameter name:

```python
fit = likelihood.fit(
    initial={"mass_0": 1.52, "width_0": 0.11},
)
```

Unknown names are errors. Run several seeded starts and compare objective
values; periodic phases and symmetry-related solutions need not have identical
coordinates.

## Project fitted components

Tags attached during model construction define coherent projections:

```python
projection = likelihood.projection(
    "signal",
    generated_mc=generated_mc,
    tags=["reference", "second"],
)

projection_weights = projection.weights(
    fit.x,
    acceptance_corrected=True,
)
```

The selected amplitudes interfere with each other. Adding separately projected
single-wave intensities generally does not reproduce the coherent projection.

## Propagate statistical uncertainty

For bootstrap uncertainty, laddu can resample each observed dataset, refit the
replica, and retain the pairing between data and fitted parameters:

```python
bootstrap = likelihood.bootstrap_fit(
    200,
    initial=fit.x,
    seed=12345,
    terminators=[ld.ganesh.MaxSteps(500)],
)
```

That pairing is important for yield and cross-section uncertainties. Inspect
fit termination, gradient size, parameter boundaries, start-to-start
stability, and bootstrap pull behavior before interpreting parameters.
