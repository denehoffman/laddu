# Efficiency, extended likelihoods, and simultaneous samples

A normalized {py:class}`laddu.NLL` determines an intensity shape. Its objective
is invariant under multiplication of the complete intensity by a constant, so
it cannot determine an absolute event yield. An
{py:class}`laddu.ExtendedNLL` instead evaluates

$$
\mathcal F_\mathrm{ext}(\boldsymbol\theta)
=\widehat{\nu}(\boldsymbol\theta)
-\sum_{i\in\mathrm{data}}w_i\log I(\Omega_i;\boldsymbol\theta),
$$

where the accepted-MC weighted sum
$\widehat{\nu}=\sum_{j\in\mathrm{accepted\ MC}}w_j I(\Omega_j)$ is interpreted
as the expected yield. The MC weights must therefore include the phase-space,
generation, exposure, and normalization factors needed to put this sum in
expected-event units.

## Efficiency and expected yields

Generated and accepted samples should start from the same thrown distribution.
In a bin $k$,

$$
\epsilon_k =
\frac{\sum_{j\in\mathrm{accepted},k}w_j}
     {\sum_{j\in\mathrm{generated},k}w_j}.
$$

Efficiency is not an extra multiplier in `ExtendedNLL` when accepted MC already
encodes the detector selection: it is present through which generated events
survive and through their weights. Applying it a second time would double-count
acceptance.

```python
extended = ld.ExtendedNLL(model, data, accepted_mc, name="period_a")
likelihood = ld.Likelihood([extended], execution=execution)
```

Include an explicit positive scale parameter in the intensity when its overall
yield is to be fitted:

```python
yield_scale = ld.parameter(
    "period_a_scale", initial=1.0, bounds=(0.0, None), scale=0.1
)
period_a_model = ld.Model(yield_scale * shape_intensity)
```

## Several datasets in one fit

Create one term per statistically independent dataset. Parameters with the same
name and compatible definitions are shared in the combined likelihood:

```python
scale_a = ld.parameter("period_a_scale", initial=1.0, bounds=(0.0, None))
scale_b = ld.parameter("period_b_scale", initial=1.0, bounds=(0.0, None))

model_a = ld.Model(scale_a * common_shape_a)
model_b = ld.Model(scale_b * common_shape_b)

term_a = ld.ExtendedNLL(model_a, data_a, accepted_mc_a, name="period_a")
term_b = ld.ExtendedNLL(model_b, data_b, accepted_mc_b, name="period_b")
joint = ld.Likelihood([term_a, term_b], execution=execution)
```

No ad-hoc factor proportional to the number of data or MC events is needed.
Each data sample contributes its own log sum, while each accepted-MC sample
contributes an integral in the units established by its weights. Different MC
sample sizes improve or degrade integration precision; they must not change the
physical integral. If unweighted MC samples of different sizes represent the
same exposure, assign weights proportional to the represented phase-space
volume divided by the number generated.

For shape-only simultaneous fits, use one `NLL` per sample. Each term then
normalizes to its own observed weighted yield, so samples with more data
naturally have more statistical influence. Use explicit term multipliers only
for a deliberate composite-likelihood convention, not to compensate for MC
sample size.

## Relative exposure and constrained scales

If the relative luminosity or exposure is known, encode it in the MC weights or
in the model:

```python
global_yield = ld.parameter("global_yield", initial=1.0, bounds=(0.0, None))
model_a = ld.Model(global_yield * common_shape_a)
model_b = ld.Model(global_yield * known_exposure_ratio * common_shape_b)
```

If the ratio is uncertain, promote it to a parameter and add an appropriate
constraint term when one is available for the uncertainty model. Ridge and
lasso penalties are regularizers, not substitutes for a correctly normalized
Gaussian or log-normal nuisance constraint.

Validate a simultaneous fit by splitting its objective into named term
projections, checking accepted-MC convergence independently in every sample,
and performing closure with the same exposure and efficiency conventions.
