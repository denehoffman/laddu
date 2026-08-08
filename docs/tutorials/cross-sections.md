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

## Construct a cross-section analysis

Continue from a fitted likelihood with named term `"signal"`:

```python
cross_section = likelihood.cross_section(
    "signal",
    generated_mc=generated_mc,
    luminosity=integrated_luminosity,
    parameters=fit.x,
)

observed = cross_section.observed_total()
acceptance = cross_section.acceptance()
corrected_yield = cross_section.corrected_yield()
```

For an `ExtendedNLL`, the same object also provides

```python
fitted = cross_section.fitted_total()
```

These methods return {py:class}`laddu.Estimate`. Without an uncertainty
ensemble only `central` is populated. `total()` remains a compatibility alias
for `observed_total()`.

## Tagged contributions

If amplitudes were tagged before model construction, select a coherent subset:

```python
reference_observed = cross_section.observed_total(tags=["reference"])
reference_acceptance = cross_section.acceptance(tags=["reference"])

# ExtendedNLL only:
reference_fitted = cross_section.fitted_total(tags=["reference"])
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

distribution = cross_section.differential(
    mass_axis,
    components={
        "reference": ["reference"],
        "second": ["second"],
    },
)
```

`distribution.data` is the bin-by-bin acceptance-corrected data estimate.
`distribution.model` is the fitted shape normalized to the total observed
yield. Tagged model components are in `distribution.components`. Values are
divided by the bin volume.

Several axes produce a flattened row-major multidimensional result:

```python
result = cross_section.differential(
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
)

model_grid = np.asarray(result.model.central).reshape(result.shape)
```

`result.axes` stores each edge array and `result.shape` stores the bin counts.

## Statistical uncertainty

Bootstrap replicas must pair each refitted parameter vector with its resampled
observed dataset. `Likelihood.bootstrap_fit` preserves this relationship:

```python
bootstrap = likelihood.bootstrap_fit(
    200,
    initial=fit.x,
    seed=12345,
    terminators=[ld.ganesh.MaxSteps(500)],
)

cross_section = likelihood.cross_section(
    "signal",
    generated_mc=generated_mc,
    luminosity=integrated_luminosity,
    parameters=fit.x,
    ensemble=bootstrap,
)

low, high = cross_section.observed_total().interval(0.68)
covariance = cross_section.differential(mass_axis).model.covariance()
```

For posterior samples, adapt the retained chain explicitly:

```python
posterior = ld.Ensemble.from_mcmc(summary, discard=1000, thin=10)
```

Evaluating bootstrap parameters against the original data loses yield
resampling and gives the wrong uncertainty for observed cross sections.

## Combining independent datasets

Build one `CrossSection` per period or selection, then pool accepted yields and
effective exposures:

```python
combined = ld.CrossSection.combine([period_a, period_b, period_c])
combined_observed = combined.observed_total()
```

This is not an arithmetic average of already-corrected cross sections. For
several decay modes of one produced state, include branching fractions as
exposure factors:

```python
all_modes = ld.CrossSection.combine(
    [mode_a, mode_b],
    factors=[branching_fraction_a, branching_fraction_b],
)
```

Factors may be floats or `Estimate` objects. Provenance-aware draws preserve
known correlations and deterministically pair unrelated ensembles.

## Low-level integrals

For a custom calculation, bypass the high-level object:

```python
integrals = likelihood.cross_section_integrals(
    "signal",
    generated_mc=generated_mc,
    tags=["reference"],
)

i_acc = integrals.accepted_integral(fit.x)
i_gen = integrals.generated_integral(fit.x)
sigma_obs = integrals.observed_cross_section(
    fit.x,
    luminosity=integrated_luminosity,
)
```

`integrals.fitted_cross_section(...)` is available for absolute-rate terms.
`Likelihood.projection` additionally exposes event-level intensities and
weights.

Cross-section uncertainties also include luminosity, finite MC statistics,
background subtraction, branching fractions, response variations, and model
dependence. The fit ensemble covers only the sources represented by its draws.
