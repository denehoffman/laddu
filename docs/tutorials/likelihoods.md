# Choosing a likelihood for event data

Likelihood choice determines which information the fit uses. The model and
datasets may be identical while the objective assigns a different role to the
overall event rate.

## Shape-only normalized likelihood

For observed weighted yield $N_w=\sum_iw_i$, `NLL` evaluates

$$
\mathcal F_\mathrm{shape}(\theta)
=-\sum_iw_i\log I(\Omega_i;\theta)
+N_w\log I_\mathrm{acc}(\theta),
$$

with accepted-MC integral

$$
I_\mathrm{acc}(\theta)
=\sum_{j\in\mathrm{accepted\ MC}}w_jI(\Omega_j;\theta).
$$

```python
shape_term = ld.NLL(
    model,
    data=data,
    accepted_mc=accepted_mc,
    name="signal",
)
shape_likelihood = ld.Likelihood([shape_term])
```

The transformation $I\mapsto cI$ leaves this objective unchanged. Use it when
the data constrain shapes and relative amplitudes but not an absolute expected
yield. One overall amplitude magnitude and phase must be fixed.

## Extended likelihood

`ExtendedNLL` retains the Poisson point-process normalization:

$$
\mathcal F_\mathrm{ext}(\theta)
=I_\mathrm{acc}(\theta)
-\sum_iw_i\log I(\Omega_i;\theta).
$$

```python
yield_scale = ld.parameter(
    "yield_scale",
    initial=1.0,
    bounds=(0.0, None),
)
extended_model = ld.Model(yield_scale * shape_intensity)

extended_term = ld.ExtendedNLL(
    extended_model,
    data=data,
    accepted_mc=accepted_mc,
    name="signal",
)
extended_likelihood = ld.Likelihood([extended_term])
```

Here the accepted-MC weights must put the integral in expected-event units.
They may include phase-space, proposal, exposure, and simulation-normalization
factors. Detector efficiency is already encoded by which generated events
survive into `accepted_mc`; multiplying by it again double-counts acceptance.

With arbitrary statistical event weights, the objective is a weighted
extended likelihood rather than a literal unweighted Poisson process. The
weight convention must be part of the analysis definition.

## Regularization terms

Penalties can be composed with either intensity likelihood:

```python
regularized = ld.Likelihood(
    [
        shape_term,
        ld.RidgePenalty(["second_magnitude"], lambda_=0.1),
    ]
)
```

Ridge adds an $L_2$ penalty and lasso adds an $L_1$ penalty to named
parameters. They are regularizers, not probability models for calibration or
exposure uncertainties. A scientifically meaningful nuisance constraint must
match the nuisance variable's actual uncertainty distribution.

## Which pathway to use

- Use `NLL` when only normalized shapes and relative amplitudes are meaningful.
- Use `ExtendedNLL` when accepted event counts carry information and MC integrals have expected-yield normalization.
- Add regularization only when its bias and interpretation are intentional.
- Use one named intensity term per statistically independent dataset; the next chapter shows how shared parameters connect those terms.
