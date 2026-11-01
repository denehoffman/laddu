# Grouped bootstrap replicas

By default, `Dataset.bootstrap` and `Likelihood.bootstrap_fit` preserve the
existing independent-row Poisson bootstrap. If several rows describe one
physical observation, configure exact integer grouping columns first:

```python
data = ld.Dataset.from_arrays(
    p4s=p4_columns,
    scalars=scalar_columns,
    columns=integer_columns,
    weights=signal_weights,
    bootstrap_groups=["run_number", "physical_event_number"],
)
likelihood = ld.Likelihood([
    ld.ExtendedNLL(model, data=data, accepted_mc=accepted_mc, name="s17")
])
ensemble = likelihood.bootstrap_fit(200, initial=central.values, seed=123)
```

Equal tuples share one Poisson multiplier, applied to the existing signed
weight. Integer columns retain their full precision, including uint64 values
above 2**53. Float expression scalars cannot serve as grouping columns. Use
identical columns and seeds to pair the same observations across selections
and models. An optional `bootstrap_namespace` separates independent samples that
reuse grouping identifiers. It is unnecessary when the grouping tuple already
identifies the observation globally, as run/event keys do in this example.
Grouped terms use the ensemble seed directly, independently of likelihood term
position. Ungrouped terms retain their established seed offsets.

The same keywords are accepted by `Dataset(source, ...)`, `Dataset.from_batches`,
`read_parquet` and `read_root`. Omit `bootstrap_groups` for the default row bootstrap;
a nonempty `bootstrap_namespace` requires explicit groups. Invalid grouping columns
are rejected during construction, before a replica is requested.

The version-one grouping hash uses length-framed UTF-8 namespace and ordered
column names, followed by exact integer values, and is deterministic across
row order, filters and batch boundaries. Group configuration changes only future
resampling. It leaves central weights and accepted/generated MC unchanged.

For a custom optimizer policy, `likelihood.bootstrap(seed=...)` returns a
prepared native replica. Fit it with the ordinary `fit` method and retain the
exact draw ID and failure status in the application ensemble. This does not
require writing replica event files. A change in requested count describes a new
ensemble; this API does not implement incremental replica reuse.

Grouped draws are a different resampling identity from historical application
hash conventions and independent-row draws. Record the grouping convention,
columns, namespace and seed in application provenance before comparing ensembles.
