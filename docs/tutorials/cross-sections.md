# Cross sections from fitted intensity

`laddu` reports the **full coherent fitted intensity** integrated over generated Monte Carlo, divided by integrated luminosity. Use an `ExtendedNLL` term so the fit determines an absolute rate; a shape-only `NLL` cannot produce a cross section.

For parameters $\theta$, let

$$
D=\sum_{i\in\mathrm{data}}w_i,\qquad
A(\theta)=\sum_{j\in\mathrm{accepted\ MC}}w_j I(\Omega_j;\theta),\qquad
G(\theta)=\sum_{j\in\mathrm{generated\ MC}}w_j I(\Omega_j;\theta).
$$

The reported result is $\sigma_\mathrm{fit}=G(\theta)/\mathcal L$. The accepted integral $A$ should be close to observed yield $D$ after an unconstrained, converged extended fit. `laddu` reports the difference between $A$ and $D$ when the fit is constrained or has not converged.

## Example: one fit, four datasets

Assume two runs each of reactions A and B. For each run, `data_*` is the
observed `Dataset`, `accepted_*` is detector-selected MC, and `generated_*`
is the corresponding generated MC. Also assume `intensity_a` and
`intensity_b` are `Expr` objects for the two reactions. They contain tagged S
and P waves and may share shape parameters. The following code gives each run
its own fitted overall scale:

```python
import laddu as ld

scale_a1 = ld.parameter("scale_a1", initial=1.0, bounds=(0.01, None))
scale_a2 = ld.parameter("scale_a2", initial=1.0, bounds=(0.01, None))
scale_b1 = ld.parameter("scale_b1", initial=1.0, bounds=(0.01, None))
scale_b2 = ld.parameter("scale_b2", initial=1.0, bounds=(0.01, None))

model_a1 = ld.Model(scale_a1 * intensity_a)
model_a2 = ld.Model(scale_a2 * intensity_a)
model_b1 = ld.Model(scale_b1 * intensity_b)
model_b2 = ld.Model(scale_b2 * intensity_b)

joint = ld.Likelihood([
    ld.ExtendedNLL(model_a1, data=data_a1, accepted_mc=accepted_a1, name="a_1"),
    ld.ExtendedNLL(model_a2, data=data_a2, accepted_mc=accepted_a2, name="a_2"),
    ld.ExtendedNLL(model_b1, data=data_b1, accepted_mc=accepted_b1, name="b_1"),
    ld.ExtendedNLL(model_b2, data=data_b2, accepted_mc=accepted_b2, name="b_2"),
])
fit = joint.fit(
    initial=joint.sample_parameters(seed=7),
    terminators=[ld.ganesh.MaxSteps(500)],
)

luminosity_a1 = ld.Luminosity(25.0, ld.AreaUnit.PICOBARN)
luminosity_a2 = ld.Luminosity(30.0, ld.AreaUnit.PICOBARN)
luminosity_b1 = ld.Luminosity(25.0, ld.AreaUnit.PICOBARN)
luminosity_b2 = ld.Luminosity(30.0, ld.AreaUnit.PICOBARN)
mass_axis = ld.Axis(ld.scalar("mass"), edges=[1.0, 1.5, 2.0])
```

Parameters declared with the same names in `intensity_a` and `intensity_b`
are shared by the joint fit. The four `scale_*` parameters are fitted
separately. Inspect `fit.converged` before using the result.

## Evaluate and inspect one term

```python
section_a1 = joint.cross_section(
    "a_1", generated_a1, luminosity_a1, fit.values
)
total = section_a1.total                    # G / luminosity, an Estimate
accepted = section_a1.accepted_integral    # A, an Estimate
generated = section_a1.generated_integral  # G, an Estimate
observed = section_a1.data_yield           # D, an Estimate
closure = section_a1.rate_closure

artifact = fit.artifact()
restored = artifact.cross_section(
    joint, "a_1", generated_mc=generated_a1, luminosity=luminosity_a1
)

integrals = joint.intensity_integrals(
    "a_1", generated_mc=generated_a1, tags=["S", "P"]
)
raw_accepted = integrals.accepted_integral(fit.values)
raw_generated = integrals.generated_integral(fit.values)
```

`section_a1` is a `CrossSection` for term `"a_1"`, evaluated with the full
joint fit. `closure.accepted_residual` is $A-D$. Its status and tolerance
identify rate mismatch without changing the cross section. `artifact` is a
`FitArtifact` that stores the fit values. The raw integrals select the S and P
waves coherently. Supplying an `Ensemble` to `joint.cross_section` also carries
paired parameter draws into the estimates.

## Differential cross sections and waves

```python
projection_a1 = section_a1.project(
    [mass_axis],
    components={"S": ["S"], "P": ["P"], "S+P": ["S", "P"]},
)
full = projection_a1.total
coherent_subset = projection_a1.components["S+P"]
```

Each bin integrates the **generated** fitted intensity and divides by its bin volume and luminosity. Selected tags are evaluated coherently, so interference is included within each selection. Single-wave contributions need not add to the full coherent total. An empty generated bin has value zero; `member_validity` reports its missing MC support separately.

## Combine runs of the same reaction

The two A terms describe the same reaction in different runs. Evaluate the
second term using the same joint fit, then pool the two results:

```python
section_a2 = joint.cross_section(
    "a_2", generated_a2, luminosity_a2, fit.values
)

reaction_a = ld.CrossSection.combine([section_a1, section_a2])
spectrum_a = reaction_a.project([mass_axis], components={"S": ["S"]})
```

Each run measures the full A reaction rate. If $G_i$ is the fitted generated integral for run $i$, the
combined result is

$$
\sigma_A=\frac{G_{A1}+G_{A2}}{\mathcal L_{A1}+\mathcal L_{A2}}.
$$

Both terms use `fit.values`, so their shared parameters and their separate
floating scales are included in $G_{A1}$ and $G_{A2}$. Projecting `reaction_a`
evaluates the same model selection in each run and pools the generated
intensity bin by bin. The runs must use compatible physical observables and
the same projection bins.

## Different reactions in one fit

The B terms use `intensity_b`, which may describe a different reaction and
share parameters with `intensity_a`. Evaluate them with the same fit and pool
the B runs separately:

```python
section_b1 = joint.cross_section(
    "b_1", generated_b1, luminosity_b1, fit.values
)
section_b2 = joint.cross_section(
    "b_2", generated_b2, luminosity_b2, fit.values
)
reaction_b = ld.CrossSection.combine([section_b1, section_b2])
spectrum_b = reaction_b.project([mass_axis], components={"S": ["S"]})
```

The two spectra retain their own cross sections even when the models share
parameters. If the reactions are exclusive channels of the same process and
the requested total is their sum, add their estimates on a common bin grid:

```python
total_spectrum = spectrum_a.total + spectrum_b.total
s_wave_spectrum = spectrum_a.components["S"] + spectrum_b.components["S"]
fitted_rate_ratio = reaction_b.total / reaction_a.total
```

`fitted_rate_ratio` is the ratio of the generated integrals **after** dividing
each by its luminosity. A scale parameter in the fit need not equal this ratio:
the models can have different phase space and intensity shapes. The ratio and
sum retain paired fit draws when the term results use the same ensemble.

## Branching fractions and a common parent cross section

If reactions A and B are observed decay channels of one parent process, their
fitted channel yields satisfy $G_i\simeq\sigma_\mathrm{parent}\mathcal L_i B_i$,
where $B_i$ is the **absolute branching fraction** for channel $i$. To pool
the two **single-run** results into one parent cross section, use those
fractions as exposure factors. Suppose independent measurements give
$B_A=0.60$ and $B_B=0.30$:

```python
branching_a = 0.60
branching_b = 0.30
parent = ld.CrossSection.combine(
    [(section_a1, branching_a), (section_b1, branching_b)]
)
```

`combine` accepts numeric factors and computes
$(G_{A1}+G_{B1})/(\mathcal L_{A1}B_A+\mathcal L_{B1}B_B)$. It accepts
single-term results such as `section_a1` and `section_b1`; it does not accept
the already pooled `reaction_a` or `reaction_b` results. If independent
measurements provide standard uncertainties of 0.02 and 0.01, pass
`Estimate` values. No random sampling is needed:

```python
parent = ld.CrossSection.combine([
    (section_a1, ld.Estimate(0.60, standard_error=0.02)),
    (section_b1, ld.Estimate(0.30, standard_error=0.01)),
])
```

For correlated branching measurements, pass their covariance matrix instead
of individual standard errors:

```python
parent = ld.CrossSection.combine(
    [(section_a1, 0.60), (section_b1, 0.30)],
    factor_covariance=[[0.0004, -0.0001], [-0.0001, 0.0001]],
)
parent_error = parent.total.error(
    data_fill=False,
    accepted_mc_fill=False,
    generated_mc_fill=False,
    branching_exposure=True,
).error
```

The matrix rows follow the member order. It describes uncertainty in the
external factors, independent of the fit. If the factors instead have paired
draws from a joint measurement, use `Estimate(value, draws=...)` for each
factor; corresponding draw positions stay paired. A fitted channel rate ratio alone
does not give absolute branching fractions. If A and B exhaust the parent
decays, `reaction_a.total + reaction_b.total` gives the parent cross section;
otherwise an absolute branching fraction is needed to infer it.
