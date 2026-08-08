# Generating model-weighted Monte Carlo

The generator from {doc}`generating-mc` can evaluate the model from
{doc}`expressions` on every proposal. This produces weighted integration data
or unweighted pseudo-data without changing the underlying phase-space
proposal.

## Weighted model events

Choose a parameter point by name:

```python
truth = {
    "mass_0": 1.50,
    "width_0": 0.12,
    "second_magnitude": 0.7,
    "second_phase": 0.9,
}

weighted_model_mc, report = generator.weighted(
    100_000,
    model=model,
    parameters=truth,
    seed=20,
)
```

Each event retains the target-to-proposal weight. Weighted samples are usually
the most efficient choice for integration and projection.

## Unweighted pseudo-data

For target-to-proposal weight $w(\Omega)$ and envelope
$W\geq\sup_\Omega w(\Omega)$, accept a proposal with

$$
P_\mathrm{accept}(\Omega)=\frac{w(\Omega)}{W}.
$$

```python
pseudo_data, report = generator.unweighted(
    10_000,
    model,
    parameters=truth,
    seed=21,
    pilot_proposals=50_000,
    safety_factor=2.0,
    grow_envelope=True,
    max_proposals=50_000_000,
)
```

A pilot envelope is useful while developing a model. For production, prefer a
validated fixed `max_weight`; if `grow_envelope=True` is used, retain the
generation report and verify that envelope updates do not reveal inadequate
proposal coverage.

## Generated, accepted, and observed samples

Keep these roles distinct:

- `generated_mc` represents the thrown phase space before detector acceptance;
- `accepted_mc` is generated MC after detector simulation and the same selection applied to data;
- `data` is the observed sample being fitted.

The accepted sample normalizes the likelihood. The generated sample is used
for acceptance corrections, model projections, and cross sections. Different
sample sizes change Monte Carlo precision, not the physical normalization
represented by their weights.
