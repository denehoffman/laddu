# Fitting a model to unbinned event data

This example defines its model and datasets in place. The small arrays make the
workflow runnable; replace them with measured data and detector-selected MC
for an analysis.

## Build a likelihood

```python
import laddu as ld
import numpy as np

def events(masses):
    masses = np.asarray(masses, dtype=float)
    return ld.Dataset.from_arrays(
        p4s={}, scalars={"mass": masses}, weights=np.ones(len(masses))
    )

data = events([1.12, 1.24, 1.39, 1.53, 1.67, 1.83])
accepted_mc = events(np.linspace(1.05, 1.95, 20))
generated_mc = events(np.linspace(1.00, 2.00, 25))

mass = ld.scalar("mass")
x = mass - 1.5
linear = ld.parameter("linear", initial=0.2, bounds=(-2.0, 2.0))
curvature = ld.parameter("curvature", initial=0.1, bounds=(-2.0, 2.0))
reference_wave = (1.0 + linear * x).tagged("reference")
second_wave = (curvature * x * x).tagged("second")
model = ld.Model((reference_wave + second_wave).norm_sqr())

term = ld.NLL(model, data=data, accepted_mc=accepted_mc, name="signal")
likelihood = ld.Likelihood([term])
initial = likelihood.sample_parameters(seed=100)
value, gradient = likelihood.value_and_gradient(initial)
```

`NLL` fits the intensity's shape. It minimizes

$$
\mathcal F(\theta)
=2\left[-\sum_{i\in\mathrm{data}}w_i\log I(\Omega_i;\theta)
+D\log A(\theta)\right],
\qquad
D=\sum_i w_i,
\qquad
A(\theta)=\sum_{j\in\mathrm{accepted\ MC}}w_j I(\Omega_j;\theta).
$$

This is the $-2\ln\mathcal L$ convention. Its overall intensity scale cancels. `likelihood.parameter_names` defines the
order of `initial` and `gradient`. Check that the initial objective and every
gradient component are finite before fitting.

## Fit and project waves

```python
fit = likelihood.fit(
    initial=initial,
    terminators=[ld.ganesh.MaxSteps(500)],
)
print(fit.converged, fit.outcome, fit.terminal_message)
print(fit.named_parameters)

projection = likelihood.projection(
    "signal", generated_mc=generated_mc, tags=["reference", "second"]
)
projection_weights = projection.weights(fit.values)
```

The two tagged waves interfere in this projection. `FitResult` exposes stable
fields such as `values`, `parameter_names`, `objective`, `converged`, and
`diagnostics`. `covariance` and `standard_errors` are available when the chosen
optimizer computed them. They include the factor of two appropriate to the
$-2\ln\mathcal L$ Hessian; the unchanged ganesh summary at
`raw_ganesh_summary` contains ganesh's inverse-Hessian covariance.

### Choose algorithms and callbacks

`fit(config=None)` keeps the existing L-BFGS-B default. Pass any of the ganesh
minimizer configurations through `ld.ganesh` to select another algorithm:

| Configuration | Algorithm |
| --- | --- |
| `LBFGSBConfig` | L-BFGS-B with native bounds |
| `ConjugateGradientConfig` | Nonlinear conjugate gradient |
| `TrustRegionConfig` | Trust region with a finite-difference Hessian |
| `AdamConfig` | Adam with full-data gradients |
| `NelderMeadConfig` | Nelder-Mead simplex |
| `CMAESConfig` | Covariance matrix adaptation evolution strategy |
| `DifferentialEvolutionConfig` | Differential evolution |
| `SimulatedAnnealingConfig` | Simulated annealing |
| `PSOConfig` | Particle swarm optimization |

Algorithm settings pass through, including custom line searches, simplex
initialization, population sizes, and error estimation modes. `seed` controls
randomized minimizers. Results retain the same `FitResult` interface.

Every ganesh Python terminator is accepted with its matching algorithm: Adam's
EMA criterion, conjugate-gradient and trust-region gradient tolerances, all
L-BFGS-B and Nelder-Mead criteria, all nine CMA-ES criteria, and simulated
annealing's temperature criterion. `MaxSteps` works with every algorithm.
Incompatible callbacks raise `TypeError` before optimization starts.

Supplied terminators are appended in order to ganesh's defaults. To replace
those defaults, set `default_callbacks=False` and supply a stopping condition:

```python
fit = likelihood.fit(
    config=ld.ganesh.ConjugateGradientConfig(
        line_search=ld.ganesh.HagerZhangLineSearch()
    ),
    default_callbacks=False,
    terminators=[
        ld.ganesh.ConjugateGradientGTerminator(eps_abs=1e-6),
        ld.ganesh.MaxSteps(500),
    ],
)
```

`ProgressObserver`, `DebugObserver`, and Python observers are accepted for all
algorithms. Python observers receive `(step, status)`; Python terminators have
the same arguments and return `True` to stop. Callback exceptions propagate
to the caller. `status` is ganesh's immutable `StatusSnapshot`.

By default, `laddu` installs likelihood parameter names, scaling, periodic
transforms, and bounds in each configuration. For direct control of those
fields, use `use_parameter_metadata=False`; the supplied ganesh configuration
then passes through unchanged. Initial vectors still use the likelihood's
free-parameter order, and objective evaluations still enforce its parameter
contract. For example:

```python
fit = likelihood.fit(
    config=ld.ganesh.LBFGSBConfig(
        bounds=[(-1.0, 1.0), (-0.5, 0.5)], error_mode="skip"
    ),
    use_parameter_metadata=False,
    terminators=[ld.ganesh.MaxSteps(500)],
)
```

`fit_restarts` and `bootstrap_fit` accept the same configurations, observers,
and callback/metadata controls and forward them to every fit.

### Stochastic Adam and ensemble sampling

`fit_stochastic` keeps Adam as its default and returns a ganesh
`MinimizationSummary`, as before. It accepts `AdamEMATerminator`, `MaxSteps`,
Python callbacks, and the same `default_callbacks` and
`use_parameter_metadata` controls as `fit`:

```python
stochastic = likelihood.fit_stochastic(
    config=ld.ganesh.AdamConfig(alpha=0.01),
    fraction=0.5,
    seed=123,
    terminators=[ld.ganesh.MaxSteps(500)],
)
```

Value and gradient requests share an event batch at each update. A gradient
consumes the cached evaluation so a subsequent update can sample a new batch
even when parameters do not move. Equal seeds reproduce the batch sequence;
`fraction=1.0` uses full data.

`sample` accepts both `AIESInit`/`AIESConfig` and `ESSInit`/`ESSConfig`. The
initialization type selects the sampler, and omitting the config selects that
sampler's defaults. All move, tuning, and chain-storage options pass through.
Both accept `AutocorrelationTerminator`, `MaxSteps`, Python callbacks,
observers, and the callback/metadata controls:

```python
walkers = likelihood.walker_positions(fit.values, 8, seed=123)
chain = likelihood.sample(
    ld.ganesh.ESSInit(walkers),
    config=ld.ganesh.ESSConfig(
        chain_storage=ld.ganesh.ChainStorage.sampled(2)
    ),
    seed=123,
    terminators=[ld.ganesh.MaxSteps(100)],
)
```

Default sampling transforms use linear scaling, preserving the target density.
When supplying nonlinear transforms through the expert path, account for the
Jacobian required by the intended density. Rust callers can already use the
complete ganesh API through `laddu::fit::ganesh` and the `FitProblem` and
`StochasticFitProblem` adapters, including custom Rust algorithms and callbacks.

To change parameter definitions, make a new model and rebuild the likelihood:

```python
fixed_model = model.with_parameters({
    "linear": ld.ParameterUpdate(fixed=0.2),
})
fixed_term = ld.NLL(
    fixed_model, data=data, accepted_mc=accepted_mc, name="signal"
)
fixed_likelihood = ld.Likelihood([fixed_term])
```

## Fit an absolute rate

A shape-only `NLL` cannot determine a cross section. Use `ExtendedNLL` when the
observed event count should constrain the fitted intensity scale:

```python
rate_scale = ld.parameter("rate_scale", initial=0.3, bounds=(0.01, None))
rate_model = ld.Model(rate_scale * (reference_wave + second_wave).norm_sqr())
rate_term = ld.ExtendedNLL(
    rate_model, data=data, accepted_mc=accepted_mc, name="signal"
)
rate_likelihood = ld.Likelihood([rate_term])
rate_fit = rate_likelihood.fit(
    initial=fit.named_parameters,
    terminators=[ld.ganesh.MaxSteps(500)],
)
luminosity = ld.Luminosity(25.0, ld.AreaUnit.PICOBARN)
section = rate_likelihood.cross_section(
    "signal", generated_mc, luminosity, rate_fit.values
)
print(section.total.central, section.rate_closure.accepted_residual)
```

`rate_scale` supplies the free overall intensity scale. The partial initial
mapping `fit.named_parameters` carries the shape fit into the extended fit;
`rate_scale` keeps its declared initial value.

`section.total` is the fitted generated-MC integral divided by `luminosity`.
The closure residual compares its accepted-MC integral with the observed yield.
See {doc}`cross-sections` for joint fits, periods, reaction channels, and
differential projections.

## Save and reuse a fit artifact

A `FitArtifact` stores results without event rows. Reconstruct the likelihood
and datasets, then bind the saved result before evaluating it:

```python
artifact = rate_fit.artifact()
artifact.save("rate-fit.laddu")

restored = ld.FitArtifact.load("rate-fit.laddu")
bound = restored.bind(rate_likelihood)
assert list(bound.values) == list(rate_fit.values)

restored_section = restored.cross_section(
    rate_likelihood,
    "signal",
    generated_mc=generated_mc,
    luminosity=luminosity,
)
```

Binding checks the likelihood structure and parameter names. The artifact can
also evaluate raw yields with `restored.yield_context(rate_likelihood,
"signal", generated_mc=generated_mc)`.

## Group fits in an analysis snapshot

An `AnalysisSnapshot` saves named fit artifacts together. An alias points to
the same artifact without duplicating it in the archive:

```python
snapshot = ld.AnalysisSnapshot()
snapshot.add_fit("fits/shape", fit.artifact())
snapshot.add_fit("fits/rate", artifact)
snapshot.alias_fit("fits/preferred", "fits/rate")
snapshot.save("analysis.laddu")

loaded = ld.AnalysisSnapshot.load("analysis.laddu")
saved_rate = loaded.fit("fits/preferred")
saved_values = saved_rate.bind(rate_likelihood).values
```

A snapshot stores fit results, not optimizer progress. You can start a **new**
fit from its saved parameter values:

```python
new_fit = rate_likelihood.fit(
    initial=saved_values,
    terminators=[ld.ganesh.MaxSteps(500)],
)
```

This restarts the optimizer at the saved point. It does not resume internal
optimizer state or recover an interrupted fit that never produced an artifact.

## Propagate statistical uncertainty

`bootstrap_fit` resamples observed events, refits each replica, and keeps the
parameter draws paired with their data replicas. For a short demonstration:

```python
bootstrap = rate_likelihood.bootstrap_fit(
    5,
    initial=rate_fit.values,
    seed=12345,
    terminators=[ld.ganesh.MaxSteps(500)],
)
artifact_with_draws = artifact.with_ensemble(bootstrap)
section_with_draws = rate_likelihood.cross_section(
    "signal", generated_mc, luminosity, rate_fit.values, ensemble=bootstrap
)
```

Use more replicas in an analysis. Check fit termination, parameter boundaries,
start-to-start stability, and bootstrap behavior before interpreting the
uncertainties.
