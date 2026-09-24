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

fitted = fit.named_parameters
assert fit.converged
print(fit.outcome, fit.terminal_message)
```

`FitResult` is laddu's stable fit façade. `values`, `parameter_names`,
`objective`, `outcome`, `converged`, and `diagnostics` do not expose optimizer
types. `covariance` and `standard_errors` are `None` when the selected method
did not compute them. Advanced optimizer diagnostics remain available through
`fit.raw_ganesh_summary`.

Create a passive in-memory record with `artifact = fit.artifact()`. Its schema,
producer version, parameter names and values, objective, terminal outcome,
diagnostics, and structural fingerprint remain inspectable without a likelihood
or any datasets. Binding is explicit: `bound = artifact.bind(likelihood)` checks
the reconstructed likelihood structure and named parameter schema, then returns
values in that likelihood's canonical order.

Persist an artifact with `artifact.save("fit.laddu")`; pass
`overwrite=True` only for an intentional replacement. `FitArtifact.load(...)`
validates the passive manifest and every binary payload before exposing fields
or allowing binding. Loading does not import code, construct a model, access a
dataset, or prepare an execution backend.

Attach uncertainty draws with `artifact.with_ensemble(ensemble)`. The archive
retains draw order, source identity, bootstrap reconstruction seed when present,
and structured failed-replica records without storing event rows. Arbitrary
replica datasets remain explicit external dependencies.

Binding accepts parameter renames only through
`bind_with_parameter_map(likelihood, mapping)`, which requires a complete
one-to-one mapping and records it on the bound state. Built-in objective terms
carry versioned identities. Custom terms must implement the Rust
`artifact_identity` contract; artifacts with unidentified terms remain
inspectable but strict binding fails.

After binding, `artifact.yield_context(...)` explicitly reconstructs central
and ensemble yield evaluation from caller-supplied likelihood and generated MC
data without running an optimizer. Reference corrections still require their
reference likelihood and datasets separately.

`AnalysisSnapshot` is a passive object graph for fit artifacts. Aliases retain
shared object identity, and the embedded object remains the same `FitArtifact`
with the standalone manifest and payload semantics. Snapshot lookup does not
bind a model, resolve datasets, prepare a backend, or evaluate the likelihood.
Save it with `snapshot.save("analysis.laddu")` and inspect it later with
`AnalysisSnapshot.load(...)`. The archive embeds each shared fit once and
reports the snapshot path if an embedded fit is corrupt. A fit artifact records
scientific fit output; an analysis snapshot groups references to such outputs.
An optimizer checkpoint instead records mutable progress for resuming an
optimization. Event datasets and replica event rows remain external to both
passive archive forms, so later yield reconstruction supplies them explicitly.

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
    fit.values,
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
    initial=fit.values,
    seed=12345,
    terminators=[ld.ganesh.MaxSteps(500)],
)
```

That pairing is important for yield and cross-section uncertainties. Inspect
fit termination, gradient size, parameter boundaries, start-to-start
stability, and bootstrap pull behavior before interpreting parameters.
