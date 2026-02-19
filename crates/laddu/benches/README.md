# Benchmarks

## Criterion JSON Report Script

`scripts/criterion_json_report.py` converts Criterion JSON-line messages into:

- summary JSON
- summary Markdown

Example usage:

```bash
cargo criterion --message-format=json \
  --bench workflow_behavior_cpu_benchmarks \
  > target/criterion/messages_cpu.jsonl

python3 crates/laddu/benches/scripts/criterion_json_report.py \
  --input target/criterion/messages_cpu.jsonl \
  --json-out target/criterion/summary/benchmark_summary_cpu.json \
  --md-out target/criterion/summary/benchmark_summary_cpu.md
```

To inspect script options:

```bash
python3 crates/laddu/benches/scripts/criterion_json_report.py --help
```

## Regression Threshold Policy

The report script supports threshold-aware regression checks:

- `--policy off|warn|fail`
- `--regression-threshold-pct <value>`
- `--improvement-threshold-pct <value>`

Example warning mode:

```bash
python3 crates/laddu/benches/scripts/criterion_json_report.py \
  --input target/criterion/messages_cpu.jsonl \
  --json-out target/criterion/summary/benchmark_summary_cpu.json \
  --md-out target/criterion/summary/benchmark_summary_cpu.md \
  --policy warn \
  --regression-threshold-pct 3.0 \
  --improvement-threshold-pct 3.0
```

Example fail mode:

```bash
python3 crates/laddu/benches/scripts/criterion_json_report.py \
  --input target/criterion/messages_cpu.jsonl \
  --json-out target/criterion/summary/benchmark_summary_cpu.json \
  --md-out target/criterion/summary/benchmark_summary_cpu.md \
  --policy fail \
  --regression-threshold-pct 3.0 \
  --improvement-threshold-pct 3.0
```

### Current CPU Thresholds And Deltas (2026-02-19)

Most recent run command:

```bash
just --justfile crates/laddu/benches/Justfile json-report
```

Configured report thresholds:

- regression threshold: `>= +3.0%`
- improvement threshold: `<= -3.0%`
- policy in this run: `off` (records status, does not fail)

Observed benchmark deltas from
`target/criterion/summary/benchmark_summary_cpu.json` (typical estimate):

- `stage_isolated_cached_value_and_expression/cached_value_fill_only`: `97.303 ns` (`NoChange`, `unchanged`)
- `stage_isolated_cached_gradient_and_expression/cached_gradient_fill_only`: `358.293 ns` (`NoChange`, `unchanged`)
- `precompute_stage_only/precompute_only`: `230626.265 ns` (`NoChange`, `unchanged`)
- `file_open/parquet_open`: `2567097.217 ns` (`NoChange`, `unchanged`)
- `file_open/root_open`: `318051.661 ns` (`NoChange`, `unchanged`)

Notes:

- These are pairwise deltas within the same run, not change-vs-previous-run deltas.
- For these benchmark IDs, Criterion change fields may be `unknown` when no prior baseline exists.

## MPI Benchmark Execution

MPI benchmark cases in `workflow_behavior_mpi_benchmarks` require launching under `mpirun`.

Build benchmark binaries:

```bash
cargo bench -p laddu --features mpi --bench workflow_behavior_mpi_benchmarks --no-run
```

Run with rank parameterization:

```bash
mpirun -n 2 cargo bench -p laddu --features mpi --bench workflow_behavior_mpi_benchmarks -- --noplot
mpirun -n 4 cargo bench -p laddu --features mpi --bench workflow_behavior_mpi_benchmarks -- --noplot
```

Machine-readable run:

```bash
mpirun -n 4 cargo criterion --features mpi --bench workflow_behavior_mpi_benchmarks --message-format=json > target/criterion/messages_mpi.jsonl
```

When multiple ranks write to stdout, post-processing should filter to rank 0 or otherwise normalize per-rank output streams before parsing.

## MPI Wall-Clock + Per-Rank RSS Metrics

Capture rank-count wall-clock and per-rank max RSS in one command:

```bash
just --justfile crates/laddu/benches/Justfile metrics-mpi rank_counts=2,4
```

Outputs:

- `target/criterion/summary/mpi_rank_metrics.json`
- `target/criterion/summary/mpi_rank_metrics.md`

Implementation details:

- Each rank executes the MPI benchmark command via `scripts/mpi_rank_probe.py`, which records child-process max RSS (kB) and elapsed time.
- Total wall-clock is measured around each `mpirun -n <ranks> ...` invocation.

## MPI Local-vs-Distributed Scaling

Capture local (single-process) and distributed scaling over representative dataset-size tiers:

```bash
just --justfile crates/laddu/benches/Justfile scaling-mpi
```

Custom tiers and ranks:

```bash
just --justfile crates/laddu/benches/Justfile scaling-mpi \
  size_tiers=small:2000,medium:5000,large:10000 \
  rank_counts=2,4
```

Outputs:

- `target/criterion/summary/mpi_local_vs_distributed_scaling.json`
- `target/criterion/summary/mpi_local_vs_distributed_scaling.md`

Implementation details:

- Size tiers are applied through `LADDU_BENCH_MAX_EVENTS` for both CPU and MPI k-matrix benchmark runs.
- Local baseline uses `workflow_behavior_cpu_benchmarks` case `kmatrix_nll_thread_scaling/value_only/1`.
- Distributed runs use `workflow_behavior_mpi_benchmarks` case `kmatrix_nll_mpi_rank_parameterized/value_only`.

## MPI Rank-Scaling Report

Generate consolidated rank-scaling bottleneck report from existing MPI artifacts:

```bash
just --justfile crates/laddu/benches/Justfile report-mpi-scaling
```

Inputs:

- `target/criterion/summary/mpi_rank_metrics.json`
- `target/criterion/summary/mpi_local_vs_distributed_scaling.json`

Outputs:

- `target/criterion/summary/mpi_rank_scaling_report.json`
- `target/criterion/summary/mpi_rank_scaling_report.md`

## File-Open Path Profiling

These commands run direct file-open pathways (no evaluator/precompute):

```bash
just --justfile crates/laddu/benches/Justfile open-parquet
just --justfile crates/laddu/benches/Justfile open-root
```

Each mode emits elapsed time, event dimensions, and process peak RSS (`VmHWM`).

## Criterion File-Open Benchmarks

Run the dedicated Criterion benchmark group for file-open pathways:

```bash
just --justfile crates/laddu/benches/Justfile bench-open
```

Emit JSON-line messages for only this benchmark group:

```bash
just --justfile crates/laddu/benches/Justfile json-open
```

The group is named `file_open` and includes:

- `parquet_open`
- `root_open`
