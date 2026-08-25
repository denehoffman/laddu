# Projection performance workloads

The projection benchmark compares the public `CrossSection::projection_set` behavior with repeated `CrossSection::differential` calls. Multiple axes in one `differential` call still mean one joint differential cross section; separate named entries in a projection set are independent cross sections that share prepared intensity work.

## Fast benchmark

Run the CodSpeed-compatible Criterion matrix with:

```console
cargo bench -p laddu --bench projection_benchmark --features likelihood
```

The matrix compares one projection-set call with one and four repeated calls for both single-member and four-period combined cross sections. Combined cases cover 20 and 200 draws plus resident and streaming data; the matrix also retains the unique-versus-duplicate-selection baseline. Criterion stores timing reports beneath `target/criterion/`. Results are engineering evidence, not an absolute release gate.

## Representative profiling workload

The manual driver defaults to one four-projection set over four run periods and 200 seeded draws, both storage policies, and serial, fixed-available, and automatic CPU thread policies:

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
