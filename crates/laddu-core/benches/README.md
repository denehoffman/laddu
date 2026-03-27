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

## Expression-IR memory metrics

Collect wall-clock, peak RSS, and an allocation proxy for expression-IR memory workloads:

```bash
python3 crates/laddu-core/benches/scripts/expression_ir_memory_metrics.py
```

Outputs:

- `target/criterion/summary/laddu_core_expression_ir_memory_metrics.json`
- `target/criterion/summary/laddu_core_expression_ir_memory_metrics.md`

To run a subset of benchmark filters:

```bash
python3 crates/laddu-core/benches/scripts/expression_ir_memory_metrics.py \
  --filters gradient_local_large/large_gradient@4096x32
```

Notes:

- `max_rss_kb` is measured from the benchmark subprocess via `ru_maxrss`.
- `allocation_proxy_bytes_per_iter` is a workload-sized logical output proxy, not a measured allocator statistic.
