# Reading, transforming, and writing data

laddu datasets are typed collections of named four-vectors, named real scalars, and one event weight. This tutorial begins in memory, then moves to lazy file-backed data.

## Construct a dataset from arrays

Four-vector arrays have shape `(events, 4)` in $(E,p_x,p_y,p_z)$ order. All columns and weights must have the same length.

```python
import laddu as ld
import numpy as np

beam = np.array([[9.0, 0.0, 0.0, 9.0], [8.5, 0.0, 0.0, 8.5]])
recoil = np.array([[1.1, 0.1, 0.0, 0.55], [1.2, -0.1, 0.1, 0.70]])
events = ld.Dataset.from_arrays(
    p4s={"beam": beam, "recoil": recoil},
    scalars={"run": np.array([12001.0, 12001.0])},
    weights=np.array([1.0, 0.8]),
)

print(events.p4_names(), events.scalar_names(), events.sum_weights())
```

### The laddu event schema

Each event has exactly three kinds of fields:

- zero or more named four-vectors, each stored as four real components;
- zero or more named real scalars;
- one real event weight, which defaults to `1.0`.

Names are schema keys, not array positions. A model that asks for
`channel.mass("X")` or `ld.Vec4.event("beam")` resolves the required components
by name when it is prepared against the dataset. All rows in one dataset share
the same schema, and duplicate names or mismatched column lengths are rejected.

Weights are multiplicative statistical weights. Selection retains them,
subsampling retains them on selected rows, and bootstrap multiplies them by
Poisson counts. Keep efficiency variables as scalar columns unless they are
already incorporated into the event weights by the analysis convention.

```{note}
Positional four-vector values use metric order `(E, px, py, pz)`, including
rows passed to `Dataset.from_arrays` and the symbolic
`Vec4(e, px, py, pz)` constructor. Dataset and file columns themselves are
named, so their physical storage order is not part of the schema.
```

Expressions are evaluated without exporting the event loop to Python. Scalar
columns are addressed by name; channel helpers provide the usual four-vector
invariants and angles for physics models:

```python
run_number = events.evaluate(ld.scalar("run"), real=True)

# With a Channel describing these columns:
mass = channel.mass("X")
m_x = events.evaluate(mass, real=True)
```

## Parquet and ROOT

Convenience readers create file-backed datasets:

```python
data = ld.read_parquet("accepted/*.parquet", chunk_size=100_000, cache="resident")
control = ld.read_root("control.root", tree="events", cache="streaming")
```

`resident` decodes each source fragment once and retains batches. It is a good default for iterative fits. `streaming` rereads batches and limits memory use, which is useful for a single pass or oversized samples.

ROOT and Parquet readers infer the laddu schema from column names. Use the
source configuration when an external file uses a nonstandard tree, component
layout, or weight column; inspect `p4_names()` and `scalar_names()` immediately
after reading unfamiliar data. A schema mismatch is caught when an expression
or model is prepared, before the fit loop starts.

Write the current transformed dataset through an explicit sink:

```python
data.write_to(ld.ParquetSink("selected.parquet", precision="f64"))
data.write_to(ld.RootSink("selected.root", tree="events", precision="f32"))
```

## Selection, resampling, and binning

Predicates remain symbolic and execute in the selected backend:

```python
selected = data.select((mass > 1.0) & (mass < 2.0), execution=execution)
small = selected.subsample(0.1, seed=4)
replica = selected.bootstrap(seed=5)

mass_bins = selected.bin_by(mass, ld.Bin.uniform(20, 1.0, 2.0), execution=execution)
for item in mass_bins:
    print(item.index, item.low, item.high, len(item.dataset))
```

Bootstrap multiplies each existing event weight by an independent Poisson(1) draw. It preserves event coordinates and is therefore convenient for uncertainty studies.

```{warning}
laddu does not infer units. Use one consistent convention—normally GeV and radians—through input, particle masses, line-shape parameters, plots, and reported results.
```
