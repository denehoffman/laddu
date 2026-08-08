# Sharing parameters across datasets

A simultaneous fit is a sum of named likelihood terms. Each term owns its
observed and accepted-MC datasets; models share a parameter when they declare
the same name with compatible bounds and metadata.

For independent datasets $d$,

$$
\mathcal F_\mathrm{joint}(\theta)
=\sum_d\mathcal F_d(\theta_d),
$$

where each $\theta_d$ contains shared and dataset-specific coordinates.

## One shape, several data-taking periods

Suppose two periods have different acceptance but share resonance parameters.
Construct each model from the same shared expressions and give nuisance
parameters distinct names:

```python
scale_a = ld.parameter("period_a_scale", initial=1.0, bounds=(0.0, None))
scale_b = ld.parameter("period_b_scale", initial=1.0, bounds=(0.0, None))

model_a = ld.Model(scale_a * shared_shape_a)
model_b = ld.Model(scale_b * shared_shape_b)

term_a = ld.ExtendedNLL(
    model_a,
    data=data_a,
    accepted_mc=accepted_mc_a,
    name="period_a",
)
term_b = ld.ExtendedNLL(
    model_b,
    data=data_b,
    accepted_mc=accepted_mc_b,
    name="period_b",
)

joint = ld.Likelihood([term_a, term_b])
```

The expressions `shared_shape_a` and `shared_shape_b` may use different event
columns or frames. Parameters such as `mass_0` and `width_0` are shared because
their declarations use the same names and definitions.

Use `NLL` instead when each period contributes only shape information. Each
normalized term then uses its own observed weighted yield and accepted-MC
normalization.

## Relative exposure

If an extended fit has a known exposure ratio $r$, encode it in the MC weights
or in the model normalization:

```python
global_scale = ld.parameter("global_scale", initial=1.0, bounds=(0.0, None))

model_a = ld.Model(global_scale * shared_shape_a)
model_b = ld.Model(global_scale * exposure_ratio * shared_shape_b)
```

Changing the number of MC events must not change the represented integral.
When an unweighted sample represents a fixed phase-space volume or exposure,
its per-event MC weight scales inversely with the number generated.

## Fit and inspect named terms

```python
fit = joint.fit(
    initial=joint.sample_parameters(seed=12),
    terminators=[ld.ganesh.MaxSteps(1000)],
)

period_a_projection = joint.projection(
    "period_a",
    generated_mc=generated_mc_a,
    tags=["reference", "second"],
)
```

Validate accepted-MC convergence separately for each term. Compare individual
and joint fits, and test whether shared parameters create tension between
datasets rather than assuming agreement by construction.
