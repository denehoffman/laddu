# Python Ingestion Benchmarks

This directory contains Python-side ingestion benchmarks for Laddu IO entrypoints.

## Run

```bash
just bench-python-ingest --repeat 5 --events 5000 --output target/benchmarks/python_ingestion_summary.json
```

Optional baseline comparison:

```bash
just bench-python-ingest \
  --repeat 5 \
  --events 5000 \
  --baseline target/benchmarks/python_ingestion_summary_baseline.json \
  --output target/benchmarks/python_ingestion_summary.json
```

## Residual Non-Zero-Copy Cases

The canonical ingestion path is now unified, but some reader paths still involve unavoidable copies:

- `from_dict`/`from_numpy` normalize columns to contiguous numeric NumPy arrays before backend ingestion.
- `from_pandas` and `from_polars` can still materialize intermediate arrays depending on optional Arrow availability and upstream dataframe layout.
- `read_root(..., backend='uproot')` and `read_amptools(...)` depend on Uproot array extraction and then normalize to canonical columns.

These residual copies are tracked by the benchmark output fields:

- `copy_diag.total_input_bytes`
- `copy_diag.non_contiguous_columns`
- `copy_diag.non_native_float_columns`
- `peak_tracemalloc_bytes`
