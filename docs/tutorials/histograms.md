# Weighted histograms

`Histogram` is a one-dimensional weighted accumulator with lower-inclusive,
upper-exclusive bins. Weighted fills retain both the signed bin contents and
the empirical sum of squared weights; `errors` is the square root of that
constituent.

```python
import laddu as ld

histogram = ld.Histogram.from_values(
    [0.2, 0.4, 0.8],
    bins=2,
    limits=(0.0, 1.0),
    weights=[2.0, -1.0, 3.0],
)

print(histogram.counts)
sumw2, underflow_sumw2, overflow_sumw2 = histogram.squared_weight_constituents()
print(sumw2)
print(histogram.errors)
```

Underflow and overflow keep their signed totals and squared-weight
constituents separately. The final upper edge belongs to overflow.

## Fill directly from a dataset

`Dataset.histogram` evaluates one real scalar expression and fills the same
`Histogram` in a single bounded traversal. Dataset event weights are included
by default, so the result follows the same signed-weight convention as
`Dataset.stats()`:

```python
x = ld.scalar("x")
histogram = dataset.histogram(x, bin_edges=[-1.0, 0.0, 1.0])
```

Set `event_weights=False` for unit base weights. An additional real scalar
weight expression multiplies whichever base weight is selected:

```python
corrected = dataset.histogram(
    x,
    bin_edges=[-1.0, 0.0, 1.0],
    weight=ld.scalar("efficiency_correction"),
)
```

The observable and optional weight are prepared together before the source is
read, then evaluated batch by batch. Resident, streaming, selected, and
explicitly chunked dataset views therefore use the same operation without
materializing a full value array. Non-finite observables or combined weights
are errors; they are never silently replaced or discarded.

## Joint histograms

`Dataset.joint_histogram` bins any non-empty ordered collection of real scalar
axes in one bounded traversal:

```python
joint = dataset.joint_histogram(
    [ld.scalar("mass"), ld.scalar("angle")],
    bin_edges=[[1.0, 1.5, 2.0], [-1.0, 0.0, 1.0]],
)
```

`joint.shape` gives the bin count for each axis. In Python, `joint.values`,
`joint.errors`, and `joint.squared_weight_constituents` are NumPy arrays with
that shape, so bins can be indexed naturally by axis. Rust and the serialized
representation retain row-major flattened storage, with the last axis varying
fastest. Every axis is lower-inclusive and upper-exclusive, including its final
upper edge.

Unlike the one-dimensional `Histogram`, a `JointHistogram` has no flow lattice.
It reports aggregate `nonfinite` and `out_of_range` diagnostics instead. If an
event has both a nonfinite coordinate or weight and an out-of-range coordinate,
the nonfinite category takes precedence. Finite signed weights excluded from
the regular bins are retained in the corresponding diagnostic weight total.

Compatible joint histograms filled from disjoint partitions can be merged.
Ordered axes and shape must match exactly, and a failed merge leaves the left
operand unchanged.

### Compatibility

`JointHistogram.to_json()` stores ordered axis edges, shape, flattened values,
squared-weight constituents, and aggregate diagnostics. Laddu validates those
relationships when loading with `JointHistogram.from_json()`; malformed or
nonfinite stored accumulators are rejected. This is a new type and does not
change the serialized or numerical contract of the existing one-dimensional
`Histogram`.

## Merge disjoint fills

Histograms filled from disjoint event partitions can be combined without
discarding their empirical uncertainties:

```python
first.merge(second)
```

The histograms must have identical bin edges, empirical-versus-manual fill
policy, and flow-constituent availability. Assigning counts or errors marks a
histogram as manual. Validation happens before the left-hand histogram is
changed, so a failed merge is atomic. Laddu combines bin contents, flow
weights, and squared-weight constituents field by field.

`Histogram` does not retain event identities. Calling `merge` therefore states
that the caller knows the fills are disjoint; merging overlapping or correlated
fills would incorrectly assume independent fill statistics. Correlated
ensemble arithmetic belongs in higher-level estimate objects.

Histograms constructed from precomputed counts keep the established
`sqrt(abs(count))` default. Supplying or assigning `errors` deliberately sets
the corresponding uncertainty-squared constituents, preserving the manual
construction workflow. Precomputed flow totals have no corresponding error
input, so their squared-weight constituents are `None` rather than an invented
value.
