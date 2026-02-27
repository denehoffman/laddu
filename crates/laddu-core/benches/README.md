# laddu-core Benchmarks

## I/O metrics capture

Collect wall-clock and peak RSS for the `open_benchmark` matrix:

```bash
python3 crates/laddu-core/benches/scripts/io_bench_metrics.py
```

Outputs:

- `target/criterion/summary/laddu_core_io_metrics.json`
- `target/criterion/summary/laddu_core_io_metrics.md`

To run a subset of benchmark filters:

```bash
python3 crates/laddu-core/benches/scripts/io_bench_metrics.py \
  --filters read_full/parquet_f32_small,read_full/root_f32_small
```

## Regression comparison

Compare candidate metrics against a saved baseline:

```bash
python3 crates/laddu-core/benches/scripts/io_bench_regression_report.py \
  --baseline target/criterion/summary/laddu_core_io_metrics.baseline.json \
  --candidate target/criterion/summary/laddu_core_io_metrics.json
```

Outputs:

- `target/criterion/summary/laddu_core_io_regression.json`
- `target/criterion/summary/laddu_core_io_regression.md`
