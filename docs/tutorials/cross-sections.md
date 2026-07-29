# From fitted intensities to cross sections

An amplitude fit normally determines shapes and relative complex couplings. An absolute cross section additionally requires integrated luminosity, efficiency, branching fractions, bin widths, and a yield convention.

For bin $k$,

$$
\left.\frac{d\sigma}{dx}\right|_k
=\frac{N_k^\mathrm{signal}}
{\mathcal L_\mathrm{int}\,\epsilon_k\,\mathcal B\,\Delta x_k}.
$$

Here $N_k^\mathrm{signal}$ may come from an extended likelihood or from acceptance-corrected fitted projection weights. Keep the convention explicit: coherent totals include interference and generally do not equal the sum of single-component yields.

## Complete four-period example

The repository includes an end-to-end $K_S^0K_S^0$ example with four fabricated
run periods. Each period has a different linear mass acceptance, event count,
and integrated luminosity. The script:

1. Generates phase-space MC and applies the period's fake detector acceptance.
2. Generates modeled, acceptance-folded pseudo-data.
3. Fits all four datasets in one likelihood with shared resonance parameters.
4. Runs paired joint Poisson-bootstrap refits.
5. Builds one `CrossSection` per period and combines their effective exposures.
6. Plots acceptance diagnostics and the combined $d\sigma/dm$ for data, the
   coherent total, and both tagged resonances.

The data error bars and all fit bands are sample standard deviations of the
propagated bootstrap ensemble. The diagnostic figure overlays the injected and
MC-measured acceptance on a second vertical axis.

```bash
just python-dev
just cross-section-example-quick cpu
just cross-section-example cpu
just cross-section-example-full jit
```

Outputs are written below `target/python-cross-section` by default, including
generated MC, accepted MC, modeled data, both figures, and a JSON fit summary.

```{literalinclude} ../../python/examples/cross_section.py
:language: python
:caption: python/examples/cross_section.py
:linenos:
```

## Preferred workflow

Create one `CrossSection` after fitting. It owns the luminosity, central
parameters, generated Monte Carlo, and (optionally) an uncertainty ensemble.
The accepted Monte Carlo and observed data come from the named likelihood term,
so these inputs cannot accidentally drift apart.

```python
cross_sections = likelihood.cross_section(
    "signal",
    generated_mc,
    luminosity=integrated_luminosity,
    parameters=fit.x,
)

sigma = cross_sections.total()
efficiency = cross_sections.acceptance()
corrected_yield = cross_sections.corrected_yield()

print(sigma.central)
```

These methods return `Estimate` objects. Without an ensemble, only `central` is
defined. With an ensemble, `draws`, `mean()`, `median()`, `std()`,
`quantile(q)`, and `interval(level)` are also available.

`total` uses the observed weight sum from the likelihood term as its accepted
yield. `corrected_yield` reports that yield after the model-weighted acceptance
correction.

### The same workflow in Rust

The implementation lives in `laddu-likelihood`; Python only converts native
objects and supplies Python-style optional arguments. Rust exposes the same
analysis types and calculations:

```rust
use std::{collections::HashMap, sync::Arc};

use laddu_likelihood::{Axis, LikelihoodResult};

fn analyze(
    likelihood: Arc<laddu_likelihood::Likelihood>,
    generated_mc: laddu_data::data::Dataset,
    mass: laddu_expr::Expr,
    fitted: Vec<f64>,
    luminosity: f64,
) -> LikelihoodResult<()> {
    let cross_sections =
        likelihood.cross_section("signal", generated_mc, luminosity, fitted)?;

    let sigma = cross_sections.total()?;
    let reference = cross_sections.total_with_tags(&["reference".into()])?;
    let components = HashMap::from([
        ("resonance_1".into(), vec!["resonance_1".into()]),
        ("resonance_2".into(), vec!["resonance_2".into()]),
    ]);
    let distribution = cross_sections.differential(
        &[Axis::new(mass, vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5])?],
        &components,
    )?;

    println!("sigma = {}", sigma.value());
    println!("reference = {}", reference.value());
    println!("model bins = {:?}", distribution.model().values());
    Ok(())
}
```

Rust also exposes `Ensemble`, `Estimate`, `BinnedEstimate`,
`DifferentialCrossSection`, `CrossSection::combine`,
`CrossSection::combine_with_factors`, covariance and interval calculations,
multidimensional axes, and provenance-aware `Estimate` arithmetic.

## Narrowing to tagged contributions

Pass `tags` to any scalar method:

```python
reference_sigma = cross_sections.total(tags=["reference"])
reference_efficiency = cross_sections.acceptance(tags=["reference"])
```

The selected tags define the numerator. The cross section retains the full
model's accepted normalization, which gives the tagged contribution its fitted
fraction of the observed yield. Interference belongs to whichever tagged
expression contains it; coherent totals generally do not equal the sum of
separately selected contributions.

## Differential cross sections

An `Axis` accepts any real laddu expression, not only invariant mass:

```python
import numpy as np

mass_axis = ld.Axis(channel.mass("X"), np.linspace(1.0, 2.0, 51))
distribution = cross_sections.differential(
    mass_axis,
    components={
        "resonance_1": ["resonance_1"],
        "resonance_2": ["resonance_2"],
    },
)

data_values = np.asarray(distribution.data.central)
model_values = np.asarray(distribution.model.central)
r1_values = np.asarray(distribution.components["resonance_1"].central)
r2_values = np.asarray(distribution.components["resonance_2"].central)
```

The component curves are separately tagged, noninterfering projections. The
coherent model still contains interference and need not equal their sum.
Values are divided by bin width, so the one-dimensional result is
$d\sigma/dx$.

Pass several axes for a multidimensional differential:

```python
result = cross_sections.differential([
    ld.Axis(channel.mass("X"), mass_edges),
    ld.Axis(cos_theta, angular_edges),
])

values = np.asarray(result.model.central).reshape(result.shape)
```

`result.axes` contains all edge arrays and `result.shape` gives the bin shape.
Values are flattened in row-major order and divided by the product of the bin
widths.

Each `BinnedEstimate` has `central`, optional `draws`, `interval(level)`, and
`covariance()`, which are sufficient for either error bars or uncertainty
bands.

## Bootstrap and MCMC uncertainty

For a Poisson bootstrap, let the likelihood build, fit, and retain the paired
replicas:

```python
bootstrap = likelihood.bootstrap_fit(
    200,
    ld.ganesh.LBFGSBConfig(history_size=10),
    initial=fit.x,
    seed=12345,
    terminators=[ld.ganesh.MaxSteps(1_000)],
)

cross_sections = likelihood.cross_section(
    "signal",
    generated_mc,
    luminosity=integrated_luminosity,
    parameters=fit.x,
    ensemble=bootstrap,
)

low, high = cross_sections.total().interval(0.68)
model_low, model_high = cross_sections.differential(mass_axis).model.interval(0.68)
```

Every bootstrap parameter draw is evaluated with its corresponding resampled
likelihood data. This pairing matters: evaluating bootstrap fit parameters
against the original observed dataset gives the wrong resampling distribution
for yields and projections.

The Rust equivalent centralizes the same resample/refit/pair operation while
letting the caller choose any optimizer:

```rust
use laddu_likelihood::Ensemble;

let bootstrap = Ensemble::bootstrap_fit(
    &likelihood,
    200,
    12_345,
    |replica, _index| fit_likelihood(replica),
)?;

let cross_sections = likelihood.cross_section_with_ensemble(
    "signal",
    generated_mc,
    integrated_luminosity,
    fitted,
    bootstrap,
)?;
```

MCMC uses the original likelihood data for every posterior draw:

```python
summary = likelihood.sample(config, init, seed=12345)
posterior = ld.Ensemble.from_mcmc(summary, discard=1_000, thin=10)

posterior_cross_sections = likelihood.cross_section(
    "signal",
    generated_mc,
    luminosity=integrated_luminosity,
    parameters=fit.x,
    ensemble=posterior,
)
```

The explicit `discard` and `thin` arguments make the chain-selection convention
visible. `Ensemble.from_arrays(values, parameter_names)` adapts custom samplers
without requiring another cross-section implementation.

## Combining datasets and decay modes

Construct one `CrossSection` per statistically independent dataset, using that
dataset's likelihood term, accepted/generated Monte Carlo, luminosity, and
ensemble. Then exposure-pool them:

```python
combined = ld.CrossSection.combine([period_a, period_b, period_c])

combined_sigma = combined.total()
combined_mass = combined.differential(
    mass_axis,
    components={"resonance_1": ["resonance_1"]},
)
```

The combination sums accepted yields and effective exposures, including each
dataset's model-weighted acceptance. It does not average already-corrected
cross sections.

For several decay modes of the same produced particle, supply the branching
fractions as exposure factors:

```python
branching = ld.Estimate(0.492, branching_draws)
all_modes = ld.CrossSection.combine(
    [mode_a, mode_b],
    factors=[branching, 0.307],
)
particle_sigma = all_modes.total(tags=["resonance_1"])
```

Factors may be positive floats or `Estimate` objects. Draws with the same
`source_id` are paired; draws from different sources are deterministically
re-paired. The same rule is used when combining ensemble-backed measurements,
which preserves known correlations while treating unrelated ensembles as
independent.

`Estimate` arithmetic applies the same provenance rule, so derived pure
quantities can be written directly:

```python
ratio = mode_a.total(tags=["resonance_1"]) / mode_b.total()
```

## Low-level diagnostics

The specialized object is the preferred analysis workflow. The lower-level
interfaces remain available for custom calculations:

- `Likelihood.cross_section_integrals` exposes aggregate accepted/generated
  integrals and normalization terms.
- `Likelihood.projection` exposes event-level intensities and weights.

Use them when the standard scalar, differential, and combination behavior is
not appropriate.

## Inspecting efficiency manually in bins

If generated and accepted samples represent the same thrown distribution,

$$\epsilon_k=\frac{\sum_{j\in\mathrm{accepted},k}w_j}{\sum_{j\in\mathrm{generated},k}w_j}.$$

```python
import numpy as np

generated_bins = generated_mc.bin_by(x, binning, execution=execution)
accepted_bins = accepted_mc.bin_by(x, binning, execution=execution)
efficiency = np.array([
    a.dataset.sum_weights() / g.dataset.sum_weights()
    for g, a in zip(generated_bins, accepted_bins, strict=True)
])
```

If generation was importance sampled, use the proper generator weights in both numerator and denominator. Evaluate uncertainty from finite MC, luminosity, background subtraction, branching fractions, model dependence, and fit statistics.

The manual calculation is useful for diagnostics and binned efficiency plots.
For fitted results, prefer `CrossSection` so that the model-weighted integrals,
normalization convention, luminosity, and uncertainty context remain coupled.

A normalized `NLL` alone cannot determine an absolute scale because
$I\mapsto cI$ cancels from its objective. Supply an independently meaningful
yield convention and luminosity before interpreting the result as an absolute
cross section.

For publication, archive the luminosity inputs, efficiency maps, response variations, selection definition, fit configuration, and code version alongside the numerical cross-section table.
