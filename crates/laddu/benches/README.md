# Benchmarks

## Criterion JSON Report Script

`scripts/criterion_json_report.py` converts Criterion JSON-line messages into:

- summary JSON
- summary Markdown

Example usage:

```bash
cargo criterion --message-format=json \
  --bench workflow_behavior_benchmarks \
  > target/criterion/messages.jsonl

python3 crates/laddu/benches/scripts/criterion_json_report.py \
  --input target/criterion/messages.jsonl \
  --json-out target/criterion/summary/benchmark_summary.json \
  --md-out target/criterion/summary/benchmark_summary.md
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
  --input target/criterion/messages.jsonl \
  --json-out target/criterion/summary/benchmark_summary.json \
  --md-out target/criterion/summary/benchmark_summary.md \
  --policy warn \
  --regression-threshold-pct 3.0 \
  --improvement-threshold-pct 3.0
```

Example fail mode:

```bash
python3 crates/laddu/benches/scripts/criterion_json_report.py \
  --input target/criterion/messages.jsonl \
  --json-out target/criterion/summary/benchmark_summary.json \
  --md-out target/criterion/summary/benchmark_summary.md \
  --policy fail \
  --regression-threshold-pct 3.0 \
  --improvement-threshold-pct 3.0
```
