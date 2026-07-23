# Generating phase-space Monte Carlo

Generation begins with a channel whose initial momenta, intermediate masses, and vertex proposals are fully specified. Consider $\gamma p\to Xp$, $X\to K_S^0K_S^0$.

```python
import laddu as ld

kaon = ld.particles.K_SHORT
channel = ld.Channel(
    "gamma p -> K_S K_S p",
    edges=[
        ld.Edge(
            "gamma", p4="gamma", particle=ld.particles.PHOTON, output=True,
            initial_momentum=ld.InitialMomentum.uniform_energy(8.0, 9.0, [0, 0, 1]),
        ),
        ld.Edge(
            "target", p4="target", particle=ld.particles.PROTON, output=True,
            initial_momentum=ld.InitialMomentum.momentum([0, 0, 0]),
        ),
        ld.Edge("X", mass_proposal=ld.MassProposal(2 * kaon.mass, 2.0)),
        ld.Edge("recoil", p4="recoil", particle=ld.particles.PROTON, output=True),
        ld.Edge("ks1", p4="ks1", particle=kaon, output=True),
        ld.Edge("ks2", p4="ks2", particle=kaon, output=True),
    ],
    vertices=[
        ld.Vertex(
            "production", incoming=["gamma", "target"], outgoing=["X", "recoil"],
            generation=ld.VertexProposal.t_exchange(
                "gamma", "X", slope=4.0, uniform_fraction=0.2,
            ),
        ),
        ld.Vertex(
            "decay", incoming=["X"], outgoing=["ks1", "ks2"],
            generation=ld.VertexProposal.isotropic(),
        ),
    ],
)
channel.validate_generation()
```

The proposal density should cover every region where the model is nonzero. It need not equal the physical distribution. Mixing a uniform tail into a forward-peaked $t$ proposal reduces the chance of missing low-probability regions.

## Weighted phase space

Omitting `model` yields the proposal/phase-space weights required for Monte Carlo integration:

```python
execution = ld.Execution("cpu", threads=8, precision="f64")
generator = ld.Generator(channel)
mc, report = generator.weighted(
    200_000, execution=execution, batch_size=4096, seed=17,
)
print(report.produced, report.sum_weights, report.maximum_weight)
mc.write_to(ld.ParquetSink("generated.parquet"))
```

Generation is deterministic for fixed inputs, seed, batch configuration, and implementation version. Treat the report as part of the dataset provenance.

## Validate the proposal

Plot every generated invariant and angle used later. Verify four-momentum conservation, thresholds, finite weights, and coverage at bin edges. Proposal defects cannot be repaired by a larger fit sample.

