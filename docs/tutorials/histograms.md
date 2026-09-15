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

Rust callers that need to share exactly the same geometry across layers can use
`laddu_physics::binning::BinningAxis`. It validates finite, strictly increasing
edges once and applies lower-inclusive, upper-exclusive internal bins. Its
`FinalUpperEdge` argument makes the outer boundary deliberate: histogram and
differential-projection filling use `Exclusive`, while bounded dataset
partitioning uses `Inclusive` so the final event is not lost. This low-level
axis is geometry and assignment policy only; it is not an accumulator or a
projection expression.

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

## Arithmetic and uncertainty provenance

Both histogram types support `+`, `-`, and multiplication by a scalar. Addition
and subtraction require identical geometry (including joint-axis order), and
scaling multiplies squared-weight constituents by the square of the factor.
The named `add`, `subtract`, and `scaled` methods provide the same operations.

Distinct histogram fills are not proof of statistical independence: they may
still evaluate the same events. Ordinary addition and subtraction therefore
preserve central values and squared-weight constituents but set the status to
`"unavailable_covariance"`; `reported_errors` is then `None`. The existing
`errors` property remains the square root of the retained constituents and is
not a reportable combined uncertainty in that state.

When independence is known outside Laddu, state it explicitly:

```python
combined = first.add(second, independent=True)
```

That assertion applies only to the two operands. It cannot recover covariance
already lost inside an operand. Paired fit ensembles and their covariance belong
to the higher-level yield and cross-section APIs, not histogram arithmetic.

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

Calling `merge` states that the caller knows the fills are disjoint. It remains
an accumulation operation, separate from arithmetic. Histogram objects do not
retain event identities, so later arithmetic remains conservative unless the
caller explicitly asserts independence.

Histograms constructed from precomputed counts keep the established
`sqrt(abs(count))` default. Supplying or assigning `errors` deliberately sets
the corresponding uncertainty-squared constituents, preserving the manual
construction workflow. Precomputed flow totals have no corresponding error
input, so their squared-weight constituents are `None` rather than an invented
value.
