# From fitted intensities to cross sections

A cross section combines a fitted intensity with luminosity, detector
acceptance, branching fractions, and a yield convention. laddu keeps these
inputs together so that tagged, differential, and uncertainty calculations use
the same normalization.

## Accepted and generated intensity integrals

For fitted parameters $\theta$, define weighted Monte Carlo integrals

$$
I_\mathrm{acc}(\theta)
=\sum_{j\in\mathrm{accepted\ MC}}w_jI(\Omega_j;\theta),
\qquad
I_\mathrm{gen}(\theta)
=\sum_{j\in\mathrm{generated\ MC}}w_jI(\Omega_j;\theta).
$$

The model-weighted acceptance is

$$
\epsilon(\theta)
=\frac{I_\mathrm{acc}(\theta)}{I_\mathrm{gen}(\theta)}.
$$

Generated and accepted samples must represent the same thrown distribution and
use compatible integration weights. Detector simulation and event selection
determine which thrown events enter the accepted sample.

## Observed and fitted normalizations

Let $N_\mathrm{obs}=\sum_iw_i$ be the observed weighted yield and $\mathcal L$
the integrated luminosity. The **observed cross section** is the
acceptance-corrected yield estimator

$$
\widehat\sigma_\mathrm{obs}(\theta)
=\frac{N_\mathrm{obs}}{\mathcal L\epsilon(\theta)}
=\frac{N_\mathrm{obs}I_\mathrm{gen}(\theta)}
{\mathcal L I_\mathrm{acc}(\theta)}.
$$

This definition works for both `NLL` and `ExtendedNLL`; the overall intensity
scale cancels.

An `ExtendedNLL` also predicts the accepted yield,
$\nu_\mathrm{acc}=I_\mathrm{acc}$. Its **fitted cross section** retains that
absolute normalization:

$$
\sigma_\mathrm{fit}(\theta)
=\frac{I_\mathrm{gen}(\theta)}{\mathcal L}.
$$

The two are related by

$$
\widehat\sigma_\mathrm{obs}
=\frac{N_\mathrm{obs}}{I_\mathrm{acc}(\theta)}
\sigma_\mathrm{fit}.
$$

They coincide when the fitted expected accepted yield equals the observed
yield. Constraints, regularization, model mismatch, or evaluation away from the
optimum can make them differ. A shape-only `NLL` cannot define
$\sigma_\mathrm{fit}$ and the fitted pathway therefore returns an error.

## Inspect scalar yields and rate closure

Build a `Yield` context when you want to inspect normalization before applying
luminosity:

```python
yield_context = likelihood.yield_context(
    "signal",
    generated_mc=generated_mc,
    parameters=fit.values,
)

selected = yield_context.selected_yield()                  # D
accepted = yield_context.accepted_fitted_yield()           # A
generated = yield_context.generated_fitted_yield()         # G
acceptance = yield_context.fitted_acceptance()              # A / G
corrected = yield_context.corrected_observed_yield()        # D G / A
closure = yield_context.rate_closure()
```

Each quantity is a separate `Estimate`; ensemble draws retain their input
ordering and source identity. `closure.status` is `"closed"`, `"failed"`, or
`"not_applicable"`, with signed, absolute, and relative residuals available
for inspection. Evaluation away from an optimum reports a failed diagnostic—it
does not rescale any quantity to manufacture closure.

For a shape-only `NLL`, fitted acceptance and corrected observed yield remain
defined because the common scale cancels. Absolute accepted/generated fitted
yields are unavailable and rate closure is explicitly not applicable.

### Project yields into bins

Use one `Axis` for a one-dimensional yield projection, or an ordered list of
axes for a joint projection. Results keep D, A, G, A/G, and DG/A together on
the same row-major bin grid:

```python
mass_axis = ld.Axis(ld.scalar("mass"), edges=[1.0, 1.5, 2.0])
angle_axis = ld.Axis(ld.scalar("angle"), edges=[-1.0, 0.0, 1.0])
projected = yield_context.projection(mass_axis)
corrected_bins = projected.corrected
validity = projected.validity
selected_histogram = projected.selected_histogram()

named = yield_context.projection_set({"mass": mass_axis, "joint": [mass_axis, angle_axis]})

# Named coherent model selections use the same bins and fitted parameters.
with_components = yield_context.projection(
    mass_axis, components={"signal": ["signal"], "background": ["background"]}
)
signal_accepted = with_components.components["signal"].accepted
signal_generated = with_components.components["signal"].generated

# An ensemble preserves the same draw order for all five quantities.
draws = projected.corrected_draws
source_id = projected.source_id
corrected_estimate = projected.corrected_estimate
if len(draws) >= 2:
    covariance = corrected_estimate.covariance()

# Select individual quantities before arithmetic.
model_yield = projected.accepted_estimate + projected.generated_estimate
```

`projection_set` preserves request order. Every entry is independent; axes
inside one entry form one joint histogram. A valid bin may have zero selected
yield. Bins without accepted or generated support have a specific validity
string and `NaN` for any undefined fitted yield, acceptance, or corrected value. For shape-only
terms, `accepted` and `generated` contain `NaN` because absolute fitted yields
are unavailable, while their ratio and the corrected selected yield remain
defined where support is positive. The `diagnostics` mapping reports nonfinite
and out-of-range coordinate counts for each sample. Histogram views copy the
central values and geometry; they do not remove the surrounding yield bundle.
The estimate accessors retain central values, paired draws, axes, and units.
`source_ids` lists the original uncertainty sources, including every source
that contributes to an exposure-combined result.
Arithmetic on individual estimates checks geometry, units, and draw counts;
central-only estimates broadcast across the other operand's draws. Draws with
the same source ID pair by index. Distinct sources use a deterministic cyclic
pairing and receive a new source ID. `has_replica_datasets` distinguishes paired
bootstrap replicas from parameter-only ensembles. Whole projection bundles do
not support arithmetic. Caller-assigned source IDs should use the lower half of
the `u64` range; generated IDs use the upper half. Reuse a returned generated ID
only to declare that draws share the same source.
Component yield projections are model-only: observed data stay with the full
selection and are never assigned to components. Their `accepted` and
`generated` arrays, validity, shape, and histogram views use the parent's bins.
Coherent interference means component yields generally do not add up to the
full fitted yield. Reordered or repeated tags select the same coherent
subexpression; each requested name remains available in `components` with
its canonical `tags`.

### Correct with an explicit reference intensity

When an analysis must reproduce a correction derived from a separate reference
model, keep that result distinct from the fitted correction:

```python
reference_corrected = yield_context.reference_corrected(
    reference_likelihood,
    "reference",
    generated_mc=reference_generated_mc,
    parameters=reference_parameters,
    ensemble=reference_ensemble,  # optional reference uncertainty
)

value = reference_corrected.value
reference_acceptance = reference_corrected.acceptance
source = reference_corrected.provenance

reference_bins = yield_context.reference_corrected_projection(
    mass_axis,
    reference_likelihood,
    "reference",
    generated_mc=reference_generated_mc,
    parameters=reference_parameters,
    ensemble=reference_ensemble,
)
corrected_bin_values = reference_bins.value.central
corrected_bin_draws = reference_bins.value.draws
```

This operation always requires an explicit reference likelihood, term, generated
sample, and parameter values. Its provenance records the reference model and
accepted/generated sample identities, parameters, and both uncertainty sources.
It applies the reference acceptance to the selected observed yield, including
signed event weights. Independent fitted and reference ensembles remain visible
as separate sources and produce derived provenance through deterministic draw
pairing. A reference correction is an analysis-reproducibility product: it does
not inherit the fitted rate-closure guarantee and is not an arbitrary rescaling
API. Its binned form uses the same axes for selected data and reference
acceptance and retains per-bin validity and both source identities.

## Convert yields with typed luminosity

Prefer converting an existing yield result when its normalization constituents
must remain inspectable:

```python
luminosity = ld.Luminosity(12.5, ld.AreaUnit.NANOBARN)
cross_section = yield_context.to_cross_section(luminosity)
differential = projected.to_cross_section(luminosity)
named_differentials = yield_context.cross_section_projection_set(
    {"mass": mass_axis, "joint": [mass_axis, angle_axis]}, luminosity
)
```

`AreaUnit` supports barn, millibarn, microbarn, nanobarn, picobarn, and
femtobarn prefixes. A
`Luminosity` is finite and positive and represents the reciprocal of its area
unit. Differential conversion divides by the joint bin measure exactly once.
Central values, draws, covariance, error-budget constituents, validity, model
components, reference provenance, and rate-closure diagnostics remain attached
to the converted result. Experiment-specific luminosity acquisition, flux
conventions, plotting units, and general unit algebra stay outside this API.

### Combine independent periods by exposure

```python
combined = ld.YieldCrossSection.combine([(first, 1.0), (second, factor)])
```

The numerator pools the underlying yields recovered from each member's typed
luminosity. The denominator sums luminosity times the positive exposure factor.
This is exposure-aware combination; adding yields, averaging cross sections,
or applying a generic rescaling does not implement the same operation.
`ExposureFactor` adds paired factor draws when their uncertainty must propagate.
Incompatible rate conventions, area prefixes, projection geometry, and component
sets are rejected before returning a result.

## Tagged contributions

If amplitudes were tagged before model construction, request coherent model
components on a yield projection:

```python
projected = yield_context.projection(
    mass_axis, components={"reference": ["reference"]}
)
reference_accepted = projected.components["reference"].accepted
reference_generated = projected.components["reference"].generated
```

For the observed pathway, the selected generated integral retains the full
model's accepted normalization:

$$
\widehat\sigma_{S,\mathrm{obs}}
=\frac{N_\mathrm{obs}I_{S,\mathrm{gen}}}
{\mathcal L I_{\mathrm{full,acc}}}.
$$

For an extended fitted prediction,

$$
\sigma_{S,\mathrm{fit}}
=\frac{I_{S,\mathrm{gen}}}{\mathcal L}.
$$

Interference belongs to whichever tagged expression retains it. A coherent
total generally differs from the sum of separately selected components.

## Differential cross sections

For a one-dimensional bin $k$ of width $\Delta x_k$, the observed estimator is

$$
\left.\frac{d\widehat\sigma_\mathrm{obs}}{dx}\right|_k
=\frac{N_k^\mathrm{obs}}
{\mathcal L\epsilon_k\Delta x_k}.
$$

An axis accepts a real laddu expression and Python or NumPy bin edges:

```python
import numpy as np

mass_axis = ld.Axis(
    generation_channel.mass("X"),
    edges=np.linspace(1.0, 2.0, 51, dtype=np.float32),
)

distribution = yield_context.projection(
    mass_axis,
    components={
        "reference": ["reference"],
        "second": ["second"],
    },
).to_cross_section(luminosity)
```

`distribution.observed` is the bin-by-bin acceptance-corrected data estimate.
`distribution.fitted` is the fitted differential cross section when the term
has an absolute rate. Tagged model components are in
`distribution.components`. Values are divided by the bin volume.

Several axes produce a flattened row-major multidimensional result:

```python
result = yield_context.projection(
    [
        ld.Axis(
            generation_channel.mass("X"),
            edges=np.linspace(1.0, 2.0, 51),
        ),
        ld.Axis(
            ld.scalar("cos_theta"),
            edges=np.linspace(-1.0, 1.0, 41),
        ),
    ]
).to_cross_section(luminosity)

model_grid = np.asarray(result.fitted.central).reshape(result.shape)
```

`result.axes` stores each edge array and `result.shape` stores the bin counts.

### Projection sets

Use a projection set when several histograms are independent rather than axes
of one joint histogram. Python accepts an insertion-ordered mapping and returns
a normal dictionary in the same order:

```python
projections = yield_context.cross_section_projection_set(
    {
        "mass": mass_axis,
        "production_angle": ld.Axis(
            ld.scalar("cos_theta"), edges=np.linspace(-1.0, 1.0, 41)
        ),
        "production_azimuth": ld.Axis(
            ld.scalar("phi"), edges=np.linspace(-np.pi, np.pi, 41)
        ),
        "decay_angles": [
            ld.Axis(ld.scalar("cos_theta_decay"), edges=np.linspace(-1.0, 1.0, 41)),
            ld.Axis(ld.scalar("phi_decay"), edges=np.linspace(-np.pi, np.pi, 41)),
        ],
    },
    luminosity,
    components={
        "reference": ["reference"],
        "second": ["second"],
    },
)

mass_distribution = projections["mass"]
decay_grid = np.asarray(projections["decay_angles"].fitted.central).reshape(
    projections["decay_angles"].shape
)
```

The component mapping applies globally to every entry. Each mapping value may
be one axis or a sequence of axes; a sequence forms one joint differential
cross section, while separate names remain independent and share prepared
event intensities.

These APIs expose the selected, accepted, generated, correction, validity,
component, and uncertainty provenance before typed luminosity conversion.

The same operation works on combined cross sections. Run-period members retain
their luminosity factors and ensemble/replica pairing while sharing prepared
intensity work across the requested projections. Reweighted replicas that share
event identity also share coordinate and bin preparation; arbitrary replica
datasets remain correct, but may require their own coordinate and bin work.

## Statistical uncertainty

Bootstrap replicas must pair each refitted parameter vector with its resampled
observed dataset. `Likelihood.bootstrap_fit` preserves this relationship:

```python
bootstrap = likelihood.bootstrap_fit(
    200,
    initial=fit.values,
    seed=12345,
    terminators=[ld.ganesh.MaxSteps(500)],
)

yield_context = likelihood.yield_context(
    "signal",
    generated_mc=generated_mc,
    parameters=fit.values,
    ensemble=bootstrap,
)
cross_section = yield_context.to_cross_section(luminosity)
mass_distribution = yield_context.projection(mass_axis).to_cross_section(luminosity)

low, high = cross_section.observed.interval(0.68)
covariance = mass_distribution.observed.covariance()
```

Integral preparations are retained by default. For a long-lived analysis with
many tagged selections or arbitrary paired replicas, set a byte limit:

```python
cross_section.configure_integral_retention(max_bytes=32 * 1024 * 1024)
cross_section.clear_integral_cache()
```

The cache evicts least-recently-used preparations when the limit is exceeded.
`max_bytes=None` disables retention for subsequent evaluations. Clearing drops
eligible cached preparations and releases their pool reservations; the original
full-model preparation remains owned by the usable `CrossSection` until it is
dropped. A preparation that exceeds the cache limit can still be evaluated
transiently. The execution memory budget remains authoritative, so a transient
preparation that cannot fit still raises the usual budget error. Native paired
bootstrap replicas share Monte Carlo event rows; arbitrary replicas whose row
identity is not proven are evaluated against their own preparations. For a
combined `CrossSection`, the byte limit and diagnostics cover all members in
aggregate.

`cross_section.diagnostics()` reports `cache_hits`, `cache_misses`,
`cache_evictions`, `cached_integrals`, and `prepared_bytes` for retained
integrals. `estimated_prepared_bytes` also includes the full-model preparation
held by the analysis after the cache is cleared. `reserved_bytes` and
`high_water_bytes` come from the host memory pools; these include other users
of the same pools, so they are useful for budget inspection rather than as an
exclusive size of this analysis. Combined diagnostics count each shared cache
and pool reservation account once. `full_requests`, `tagged_requests`,
`central_requests`, `shared_bootstrap_requests`, and
`arbitrary_replica_requests` show which evaluation paths ran. Request counts
include the initial full-model preparation. The high-water value is historical
for the pool and does not decrease when a cache entry is evicted or dropped.

For posterior samples, adapt the retained chain explicitly:

```python
posterior = ld.Ensemble.from_mcmc(summary, discard=1000, thin=10)
```

Evaluating bootstrap parameters against the original data loses yield
resampling and gives the wrong uncertainty for observed cross sections.

## Combining independent datasets

Convert one yield per period or selection with typed luminosity, then pool
underlying yields and effective exposures:

```python
combined = ld.YieldCrossSection.combine(
    [(period_a, 1.0), (period_b, 1.0), (period_c, 1.0)]
)
combined_observed = combined.observed
```

This is not an arithmetic average of already-corrected cross sections. For
several decay modes of one produced state, include branching fractions as
exposure factors:

```python
all_modes = ld.YieldCrossSection.combine_with_factors(
    [(mode_a, branching_fraction_a), (mode_b, branching_fraction_b)]
)
```

Factors are explicit `ExposureFactor` objects when they carry uncertainty.
Provenance-aware draws preserve known correlations and deterministically pair
unrelated ensembles.

## Low-level integrals

For a custom calculation, bypass the high-level object:

```python
integrals = likelihood.cross_section_integrals(
    "signal",
    generated_mc=generated_mc,
    tags=["reference"],
)

i_acc = integrals.accepted_integral(fit.values)
i_gen = integrals.generated_integral(fit.values)
sigma_obs = integrals.observed_cross_section(
    fit.values,
    luminosity=integrated_luminosity,
)
```

`integrals.fitted_cross_section(...)` is available for absolute-rate terms.
`Likelihood.projection` additionally exposes event-level intensities and
weights.

Cross-section uncertainties also include luminosity, finite MC statistics,
background subtraction, branching fractions, response variations, and model
dependence. The fit ensemble covers only the sources represented by its draws.
