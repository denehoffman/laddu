# Projection performance workloads

The projection baseline measures the existing public `CrossSection::differential` behavior. Multiple axes in one call still mean one joint differential cross section; these workloads make one or four separate calls to represent independent projections.

## Fast benchmark

Run the CodSpeed-compatible Criterion matrix with:

```console
cargo bench -p laddu --bench projection_benchmark --features likelihood
```

The matrix covers one and four separate calls, 20 and 200 deterministic ensemble draws, single-member and four-period combined cross sections, resident and streaming prepared data, and unique versus duplicate canonical selections. Criterion stores timing reports beneath `target/criterion/`. Results are engineering evidence, not an absolute release gate.

## Representative profiling workload

The manual driver defaults to four run periods, four projection calls, 200 seeded draws, both storage policies, and serial, fixed-available, and automatic CPU thread policies:

```console
cargo run -p laddu --example projection_profile --features likelihood --release
```

Use smaller cases while iterating, or select one policy:

```console
cargo run -p laddu --example projection_profile --features likelihood --release -- \
  --events 1000 --draws 20 --storage resident --threads fixed:4
```

The driver writes machine-specific timings only to `target/projection-profile/summary.csv` (or the equivalent directory under `CARGO_TARGET_DIR`). Store profiler and flame-graph output beside it, for example:

```console
cargo flamegraph -p laddu --example projection_profile --features likelihood \
  --output target/projection-profile/flamegraph.svg -- \
  --storage resident --threads serial
```

Do not commit generated timings, profiles, flame graphs, or machine-specific comparisons. The deterministic definitions and public behavior tests are the committed baseline.
