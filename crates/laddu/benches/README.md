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
