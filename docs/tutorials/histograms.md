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
