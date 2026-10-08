# Reading, transforming, and writing event data

A {py:class}`laddu.Dataset` is a typed collection of named four-vectors,
named real scalars, exact integer row-data columns, and one statistical weight
per event. Expressions request four-vectors and real scalars by name. Integer
columns carry identifiers and other payload independently of expressions.

## Construct data from NumPy arrays

Four-vectors have shape `(events, 4)` in $(E,p_x,p_y,p_z)$ order. Scalar
columns and weights have shape `(events,)`. Both `float32` and `float64` arrays
are accepted and converted to `laddu`'s internal real representation.

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
    scalars={"polarization": np.array([0.4, 0.6], dtype=np.float32)},
    columns={
        "run": np.array([12001, 12001], dtype=np.uint32),
        "event_id": np.array([2**63, 2**63 + 1], dtype=np.uint64),
    },
    weights=np.array([1.0, 0.8], dtype=np.float32),
)
```

All columns must contain the same number of events. Duplicate names, invalid
four-vector shapes, and length mismatches fail at construction.

`events.column("run")` and `events.column("event_id")` return exact integer
values and their original dtypes; `events.column("polarization")` returns the
stored float64 scalar values. The `columns` input infers dtypes from its arrays;
the `scalars` input converts numeric values to floating-point storage.
`column_names()` lists real scalars followed by integer columns in schema order.
Arrays are copied on construction and exported as independent, initially
read-only NumPy arrays. Integer names must be distinct from scalar, logical
four-vector, physical component, and weight names.

All eight signed and unsigned 8-, 16-, 32-, and 64-bit integer dtypes are
supported. Schema declarations accept canonical names (`"uint64"`), Rust names
(`"u64"`), NumPy dtype objects, and NumPy integer scalar classes. String `"u8"`
means **uint8**, while `np.dtype("u8")` means **uint64**, following NumPy's
byte-count shorthand. Floats, booleans, strings, objects, and nullable integer
payload are not accepted as new row-data columns. These columns survive views
and I/O without becoming expression inputs or changing bootstrap row identity.

```{note}
`laddu` does not infer units. Use one convention—normally GeV and radians—for
input data, particle masses, model parameters, bin edges, and reported results.
```

## Read file-backed data

Convenience readers infer the `laddu` schema and create lazy datasets:

```python
data = ld.read_parquet("accepted/*.parquet")
control = ld.read_root("control.root", tree="events")
```

No dtype declarations are needed for ordinary reads: native readers discover
supported integer columns from the file metadata and preserve their widths and
signedness. Inspect `data.schema.columns` to see the discovered dtypes.

Optionally supply a schema to project fields and assert their expected types:

```python
ids_only = ld.read_parquet(
    "accepted/*.parquet", schema=ld.Schema(columns={"event_id": "u64"})
)
```

This reads only `event_id` and requires it to already be uint64. It never casts
integer columns, including signed-to-unsigned conversions. Unrelated nullable
columns are excluded by projection. Selected integer nulls, missing fields,
and dtype differences across files raise errors.
Write `precision` controls float fields; integer dtypes and values remain exact.

Use `p4_names()`, `scalar_names()`, and `column_names()` to check unfamiliar files. A model-schema
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

Expressions keep event loops inside `laddu`. A scalar column is addressed by
name, while reaction-channel helpers introduced in {doc}`quantum-numbers`
construct invariant masses and angles from named four-vectors.

```python
polarization = ld.scalar("polarization")
polarization_values = events.evaluate(polarization, real=True)

selected = events.select((polarization >= 0.3) & (polarization < 0.7))
small = selected.subsample(0.1, seed=4)
replica = selected.bootstrap(seed=5)
```

Pass a mapping to evaluate several named expressions in one bounded source
traversal. Returned arrays follow mapping order and align with the dataset's
weights and other separately collected columns:

```python
angles = events.evaluate({"theta": theta, "phi": phi, "P": P, "Phi": Phi})
```

Each entry must be a scalar expression without free parameters. `real=True`
requires real-valued expressions and returns `float64` arrays; it rejects complex
expressions rather than discarding their imaginary parts. The default returns
`complex128` arrays. Execution precision governs computation, while these output
dtypes remain unchanged. An empty mapping returns `{}` without reading events.
Reusable sources must replay the same ordered rows, even if batch boundaries
change, so arrays collected in separate calls remain aligned.

`bootstrap` multiplies each existing event weight by an independent
Poisson$(1)$ draw while preserving event coordinates. `subsample` selects a
reproducible subset without changing retained weights.

## Inspect weight statistics

`stats()` evaluates the current immutable view once and caches the result for
subsequent calls, including after filters, transformations, and chunking:

```python
summary = selected.stats()
print(summary.events, summary.sum_weights)
print(summary.sum_squared_weights, summary.effective_entries)
print(summary.positive_weights, summary.negative_weights)
```

`sum_weights` is signed. The positive and negative accessors expose the
separated signed contributions, while `sum_squared_weights` is nonnegative
and is the quantity used to derive weighted fill uncertainties. Effective
entries are `(sum_weights ** 2) / sum_squared_weights`; they are `None` for an
empty or all-zero-weight view because the denominator is not positive. These
diagnostics are distinct from any later yield error budget.

## Bin event data

A bin specification can be uniform or use explicit Python/NumPy edges:

```python
uniform = ld.Bin.uniform(20, low=0.0, high=1.0)
explicit = ld.Bin(np.linspace(0.0, 1.0, 21, dtype=np.float32))

polarization_bins = selected.bin_by(polarization, bins=explicit)
first_bin_data = polarization_bins[0].dataset
```

The result includes every interval in edge order, including empty bins. Each
item retains its index, limits, and dataset.

## Write transformed data

Sinks make output format and precision explicit:

```python
selected.write_to(ld.ParquetSink("selected.parquet", precision="f64"))
selected.write_to(ld.RootSink("selected.root", tree="events", precision="f32"))
```

Both built-in and external writers return a `WriteResult` with output `paths`,
per-file `counts`, and total `events`. Writers consume the current logical view,
including selections and effective bootstrap weights.

## Stream data into an external library

`Dataset.batches()` exposes the same logical traversal without reconstructing
columns through separate expression evaluations:

```python
with selected.batches(chunk_size=4096) as batches:
    for batch in batches:
        beam = batch.p4s["beam"]       # float64, shape (N, 4): E, px, py, pz
        polarization = batch.scalars["polarization"]  # float64, shape (N,)
        run = batch.columns["run"]    # uint32, shape (N,)
        weights = batch.weights       # explicit effective weights, shape (N,)
```

`selected.schema` describes these exported columns. Export always makes weights
explicit, including unit weights. Batches own their data and remain valid after
the iterator advances; NumPy arrays and column mappings are read-only. Use a
context manager or `close()` when abandoning a traversal. An exhausted iterator
closes automatically.

## Define an external file format

`laddu.io.FormatSpec` binds three ordinary Python callbacks. There is no format
registry and no need to modify `laddu`. The file does not need to be columnar:
decode ASCII records, ROOT array branches, or nested objects into canonical
event batches inside the reader, and reconstruct the format inside the writer.

This complete example stores nested momentum records as JSON lines:

```python
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from laddu.io import EventBatch, FormatSpec, Schema, SourceInfo


@dataclass(frozen=True)
class RecordOptions:
    particle: str = "beam"
    scale: float = 1.0


def describe(source, *, options):
    return SourceInfo(Schema(p4s=[options.particle], weights=True))


def read_batches(source, *, options, plan):
    # A generator opens a fresh traversal on every call and owns its resources.
    with Path(source).open() as file:
        for line in file:
            record = json.loads(line)
            yield EventBatch(
                p4s={options.particle: np.array([record["momentum"]]) * options.scale},
                weights=np.array([record["weight"]], dtype=float),
            )


def write_batches(target, batches, *, options, schema, plan):
    with Path(target).open("w") as file:
        for batch in batches:
            momenta = batch.p4s[options.particle] / options.scale
            for momentum, weight in zip(momenta, batch.weights, strict=True):
                file.write(json.dumps({
                    "momentum": momentum.tolist(), "weight": float(weight),
                }) + "\n")


records = FormatSpec(
    name="momentum-records",
    describe=describe,
    read_batches=read_batches,
    write_batches=write_batches,
    extension="jsonl",
)

options = RecordOptions(scale=0.001)  # file units -> analysis units
records.write(events, "events.jsonl", options=options)
restored = records.read("events.jsonl", options=options, cache="streaming")
```

The reader must yield batches no larger than `plan.chunk_size` and obey the
declared schema. Four-vector and scalar input arrays may be float32 or float64;
`EventBatch` copies them into immutable float64 storage. Missing weights mean
unit weights. Use `EventBatch(length=N)` for events with no columns. Name order
is resolved against `Schema`, independent of mapping insertion order.

For exact payload, declare `Schema(columns={"event_id": "u64"})` and yield
`EventBatch(columns={"event_id": ids})`, or include a `"columns"` mapping in a
batch dictionary passed through `Dataset.from_batches`. Exported
`batch.columns` is a read-only mapping of independent arrays; `schema.columns`
contains canonical dtype names. Every batch must match the declared names and
integer dtypes exactly. Writers receive these same declarations and arrays,
including generated index columns. Readers must replay identical ordered rows
and contents on repeated reads; batch boundaries may vary. This requirement
keeps separate column, weight, and expression calls aligned.

`SourceInfo.length` is an optional **global** count, not a rank-local count.
Metadata callbacks should inspect headers or cheap format metadata rather than
materialize the file. Options may be any Python object; the format author owns
their types, defaults, and validation.

Generated type stubs check callback return types, native partition strategy names,
and memory settings. Callbacks can annotate their own source and option types,
including format-specific dataclasses. The generic spec accepts these values as
`object`; publish typed wrappers such as `read_records` below when callers should
be checked against your format's particular option type. Callback keyword
arguments follow the contracts shown above; the `Callable[..., ...]` hints do
not check their names.

The writer owns headers, encoding, flushing, and cleanup, and must consume the
entire iterator. Returning early is an error. Empty datasets still invoke the
writer with their schema. Exceptions identify the format/operation and preserve
the local Python cause when crossing the source seam. Partial outputs may remain
after failure.

Reader-only and writer-only specs are supported. Bound adapters also compose
with the existing interface:

```python
data = ld.Dataset(records.source("events.jsonl", options=options))
result = data.write_to(records.sink("copy.jsonl", options=options))
```

An external package can publish ordinary wrappers with its preferred signatures:

```python
def read_records(path, *, particle="beam", scale=1.0):
    return records.read(path, options=RecordOptions(particle, scale))
```

`ld.io.convert(source, target, reader=..., writer=..., read_options=...,
write_options=...)` connects two specs using streaming caching by default.

## Distributed format conversion

Pass an `Execution` explicitly to reads, batch iteration, conversion, or writing
when choosing a partitioning policy. Otherwise the installed backend supplies
the same automatic execution defaults as evaluation. Calling `read` does not
persist that execution choice into subsequent operations.

```python
execution = ld.Execution("cpu", partitioning="rows")
records.write(data, "output.jsonl", options=options,
              execution=execution, output="single", chunk_size=4096)
```

All ranks must participate in distributed reads that inspect metadata and in
distributed writes. Metadata/schema and output settings must agree across ranks.
Callbacks never perform MPI collectives themselves.

The default reader receives a serial plan (`rank=0`, `nranks=1`) on every rank.
`laddu` partitions its output by global event position **before** dataset
transformations. Row partitioning keeps positions with `position % nranks ==
rank`. Contiguous partitioning assigns `[N * rank // nranks, N * (rank + 1) //
nranks)`; if the count is unknown, it first counts a fresh serial stream. This
fallback supports sequential formats but repeats I/O on each rank.

An efficient reader explicitly declares native strategies:

```python
fast_records = FormatSpec(
    name="indexed-records", describe=describe,
    read_batches=read_indexed_batches,
    native_partitioning={"contiguous", "rows"},
)
```

For a declared strategy, the callback receives the real rank plan and must yield
only that rank's assigned events, in source order. `laddu` does not partition
them again. The partition formulas above also define native contiguous/row
assignment, making ownership independent of batch size. `laddu` supplies stable
row identities for these strategies so bootstrap and subsampling agree with
serial traversal. Native `file_groups` readers must attach globally unique
source positions using `EventBatch(..., row_ids=[...])`; file groups have no
generic fallback.

Test native assignment locally on small fixtures, without launching MPI:

```python
fast_records.check_partitioning("fixture.jsonl", options=options,
                                nranks=3, chunk_size=2)
```

This helper compares every declared strategy with serial traversal, detecting
overlap, omissions, and changed event values. It deliberately materializes the
fixture. Repeat it with different rank counts and chunk sizes.

Output modes are shared by custom writers and `Dataset.write_to`:

- `auto`: one file locally, rank-suffixed files under MPI.
- `sharded`: each rank writes its local stream to its resolved path. Paths
  without an extension denote a directory containing `part-rank...` files.
- `single`: rank zero consumes a bounded stream from all ranks in rank order.
  Local order is preserved; row/file-group partitioning can differ from serial
  global order. Other ranks provide batches on demand and never open the target.

`WriteResult.paths` and `counts` describe the successfully written files and are
identical on all ranks. Distributed output targets are filesystem paths. Output
failure or premature writer return cancels the transport and closes readers;
an incomplete file is not returned as a successful result.

The next chapter builds particles and a reaction channel whose edge names match
the dataset schema.
