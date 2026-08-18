# Generating model-independent Monte Carlo

Phase-space Monte Carlo provides integration events before a physical intensity
is chosen. Generation requires the reaction topology from
{doc}`quantum-numbers` plus proposal distributions for initial momenta,
intermediate masses, and vertices.

## Add generation proposals

For $\gamma p\to Xp$, $X\to K_S^0K_S^0$, the proposal may sample the photon
energy, the intermediate mass, a forward-peaked production transfer, and an
isotropic decay:

```python
generation_channel = ld.Channel(
    "gamma p -> K_S K_S p",
    edges=[
        ld.Edge(
            "gamma",
            p4="gamma",
            particle=ld.particles.PHOTON,
            output=True,
            initial_momentum=ld.InitialMomentum.uniform_energy(
                low=8.0,
                high=9.0,
                direction=[0.0, 0.0, 1.0],
            ),
        ),
        ld.Edge(
            "target",
            p4="target",
            particle=ld.particles.PROTON,
            output=True,
            initial_momentum=ld.InitialMomentum.momentum([0.0, 0.0, 0.0]),
        ),
        ld.Edge("X", mass_proposal=ld.MassProposal(0.995, high=2.0)),
        ld.Edge(
            "recoil", p4="recoil", particle=ld.particles.PROTON, output=True
        ),
        ld.Edge("ks1", p4="ks1", particle=ld.particles.K_SHORT, output=True),
        ld.Edge("ks2", p4="ks2", particle=ld.particles.K_SHORT, output=True),
    ],
    vertices=[
        ld.Vertex(
            "production",
            incoming=["gamma", "target"],
            outgoing=["X", "recoil"],
            generation=ld.VertexProposal.t_exchange(
                incoming="gamma",
                outgoing="X",
                slope=4.0,
                uniform_fraction=0.2,
            ),
        ),
        ld.Vertex(
            "decay",
            incoming=["X"],
            outgoing=["ks1", "ks2"],
            generation=ld.VertexProposal.isotropic(),
        ),
    ],
)

generation_channel.validate_generation()
generator = ld.Generator(generation_channel)
```

The proposal density $q(\Omega)$ must cover every region where a later model
can be nonzero. It need not resemble the physical distribution. A uniform
mixture in an otherwise forward-peaked proposal protects low-probability
regions from receiving no samples.

## Add generated scalar columns

Named scalar sources are sampled alongside the event four-momenta. They can
represent auxiliary quantities such as beam polarization and are available to
models through ordinary scalar expressions:

```python
generator = ld.Generator(
    generation_channel,
    scalars={
        "pol_magnitude": ld.ScalarSource.uniform(0.2, 0.3),
        "pol_angle": ld.ScalarSource.fixed(0.0),
    },
)

pol_magnitude = ld.scalar("pol_magnitude")
pol_angle = ld.scalar("pol_angle")
```

`ScalarSource.histogram(histogram)` supplies a piecewise-constant alternative.
The same sources serialize to tagged JSON and expose JSON Schema metadata for
downstream pipeline configuration formats.

## Produce weighted phase space

Without a model, `weighted` returns the phase-space/proposal weights required
for Monte Carlo integration:

```python
generated_mc, report = generator.weighted(
    200_000,
    seed=17,
)

generated_mc.write_to(ld.ParquetSink("generated.parquet"))
```

Generation is reproducible for fixed inputs and seed. Event chunking and memory
budgets are execution details deferred to {doc}`execution`.

## Validate before modeling

Evaluate the invariants that will enter the model and check their support:

```python
mass_x = generation_channel.mass("X")

mass_values = generated_mc.evaluate(mass_x, real=True)
```

Verify thresholds, four-momentum conservation, finite weights, coverage near
analysis boundaries, and adequate effective sample size. A physical model can
reweight a sound proposal; it cannot repair missing phase space.
