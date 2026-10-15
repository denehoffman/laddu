# Choosing a likelihood for event data

Likelihood choice determines which information the fit uses. The model and
datasets may be identical while the objective assigns a different role to the
overall event rate.

All built-in likelihood objectives use $Q=-2\ln\mathcal L$. The existing
`nll`, `value`, and `value_and_gradient` APIs retain their names and return this
convention, as do fit objectives and stochastic evaluations. Values and gradients
are twice those from the former $-\ln\mathcal L$ convention; likelihood optima
are unchanged. Parameter-independent normalization constants remain omitted.

## Shape-only normalized likelihood

For observed weighted yield $N_w=\sum_iw_i$, `NLL` evaluates

$$
\mathcal F_\mathrm{shape}(\theta)
=2\left[-\sum_iw_i\log I(\Omega_i;\theta)
+N_w\log I_\mathrm{acc}(\theta)\right],
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
=2\left[I_\mathrm{acc}(\theta)
-\sum_iw_i\log I(\Omega_i;\theta)\right].
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
        ld.RidgePenalty(["second magnitude"], lambda_=0.1),
    ]
)
```

Ridge adds $2\lambda\sum_k\theta_k^2$ and lasso adds
$2\lambda\sum_k|\theta_k|$ to named parameters. Their strengths relative to
the likelihood are unchanged from the former convention. They are regularizers,
not probability models for calibration or exposure uncertainties. A scientifically meaningful nuisance constraint must
match the nuisance variable's actual uncertainty distribution.

## Sampling and uncertainty

Ensemble samplers use $\ln\mathcal L=-Q/2$, so changing the objective convention
does not sharpen the target density. `FitResult.covariance` uses
$2[\nabla^2Q]^{-1}$ (or the corresponding pseudoinverse), and `standard_errors`
are its diagonal square roots. This preserves the physical uncertainty scale.
The dependency-specific `raw_ganesh_summary` retains ganesh's inverse-Hessian
covariance, which is half the covariance in `FitResult` under this convention.
Weighted data still require the analysis's intended uncertainty treatment.

Custom Rust `LikelihoodTerm` implementations must supply their values and
gradients in $-2\ln\mathcal L$ units. The generic Rust `FitProblem` sampler
adapter interprets any supplied objective using $\ln\mathcal L=-Q/2$.

Built-in likelihood and penalty identities have changed with their numerical
convention. Saved artifacts from the old convention remain readable as historical
results, but fail structural binding against these new likelihoods; their stored
objective values are preserved. Refit to create an artifact in the new convention.

## Which pathway to use

- Use `NLL` when only normalized shapes and relative amplitudes are meaningful.
- Use `ExtendedNLL` when accepted event counts carry information and MC integrals have expected-yield normalization.
- Add regularization only when its bias and interpretation are intentional.
- Use one named intensity term per statistically independent dataset; the next chapter shows how shared parameters connect those terms.
