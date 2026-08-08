# Reading, transforming, and writing event data

A laddu {py:class}`laddu.Dataset` is a typed collection of named four-vectors,
named real scalars, and one statistical weight per event. Names form the event
schema: later expressions request columns by name rather than by position.

## Construct data from NumPy arrays

Four-vectors have shape `(events, 4)` in $(E,p_x,p_y,p_z)$ order. Scalar
columns and weights have shape `(events,)`. Both `float32` and `float64` arrays
are accepted and converted to laddu's internal real representation.

```python
import laddu as ld
import numpy as np

events = ld.Dataset.from_arrays(
    p4s={
        "beam": np.array(
            [[9.0, 0.0, 0.0, 9.0], [8.5, 0.0, 0.0, 8.5]],
            dtype=np.float32,
        ),
        "recoil": np.array(
            [[1.1, 0.1, 0.0, 0.55], [1.2, -0.1, 0.1, 0.70]],
        ),
    },
    scalars={"run": np.array([12001, 12001], dtype=np.float32)},
    weights=np.array([1.0, 0.8], dtype=np.float32),
)
```

All columns must contain the same number of events. Duplicate names, invalid
four-vector shapes, and length mismatches fail at construction.

```{note}
laddu does not infer units. Use one convention—normally GeV and radians—for
input data, particle masses, model parameters, bin edges, and reported results.
```

## Read file-backed data

Convenience readers infer the laddu schema and create lazy datasets:

```python
data = ld.read_parquet("accepted/*.parquet")
control = ld.read_root("control.root", tree="events")
```

Use `p4_names()` and `scalar_names()` to check unfamiliar files. A model-schema
mismatch is reported when an expression is prepared, before optimization.

Memory and cache controls are optional operational choices:

```python
data = ld.read_parquet(
    "accepted/*.parquet",
    memory="2 GiB",
    cache="fastest",
)
```

`fastest` caches decoded data when it fits and otherwise streams chunks.
`resident` requires a fully cached dataset; `streaming` requires rereads.

## Evaluate and transform

Expressions keep event loops inside laddu. A scalar column is addressed by
name, while reaction-channel helpers introduced in {doc}`quantum-numbers`
construct invariant masses and angles from named four-vectors.

```python
run = ld.scalar("run")
run_values = events.evaluate(run, real=True)

selected = events.select((run >= 12000) & (run < 13000))
small = selected.subsample(0.1, seed=4)
replica = selected.bootstrap(seed=5)
```

`bootstrap` multiplies each existing event weight by an independent
Poisson$(1)$ draw while preserving event coordinates. `subsample` selects a
reproducible subset without changing retained weights.

## Bin event data

A bin specification can be uniform or use explicit Python/NumPy edges:

```python
uniform = ld.Bin.uniform(20, low=12000.0, high=13000.0)
explicit = ld.Bin(np.linspace(12000.0, 13000.0, 21, dtype=np.float32))

run_bins = selected.bin_by(run, bins=explicit)
first_bin_data = run_bins[0].dataset
```

The result includes every interval in edge order, including empty bins. Each
item retains its index, limits, and dataset.

## Write transformed data

Sinks make output format and precision explicit:

```python
selected.write_to(ld.ParquetSink("selected.parquet", precision="f64"))
selected.write_to(ld.RootSink("selected.root", tree="events", precision="f32"))
```

The next chapter builds particles and a reaction channel whose edge names match
the dataset schema.
