# CPU baseline — 2026-10-02

Collected before CPU optimization, at revision `d1c1295105677aa4b366ba3b894f0c1800376e59` with the new benchmark tooling uncommitted.

Machine: AMD Ryzen 7 3700X 8-Core Processor; Linux-6.18.54-x86_64-with-glibc2.42. Each routine case ran three times with four threads and three times with one thread, sequentially in fresh processes.

Build: release, `--no-default-features --features fit`, disabled JIT, CPU f64, automatic normalization. Compiler: `rustc 1.99.0-nightly (77cf889bc 2026-07-12)`.

All 48 routine processes completed below the 28,000,000,000-byte RSS watchdog threshold. Four-thread results below use median total time and maximum process RSS across three repetitions.

| Size | Workflow | Total seconds | Peak RSS MB |
| --- | --- | ---: | ---: |
| small | likelihood-5 | 0.914 | 14.21 |
| small | likelihood-12 | 2.454 | 26.62 |
| small | bootstrap | 3.992 | 28.92 |
| small | parquet | 0.072 | 14.41 |
| reference | likelihood-5 | 1.777 | 33.43 |
| reference | likelihood-12 | 7.138 | 37.95 |
| reference | bootstrap | 14.200 | 161.59 |
| reference | parquet | 0.345 | 64.74 |

Reference bootstrap stages (four threads, medians):

| Stage | Seconds |
| --- | ---: |
| bootstrap_fits | 2.152622 |
| bounded_fit | 0.215879 |
| component_projections | 3.971156 |
| cross_section_total | 0.916674 |
| fixed_gradient | 0.048034 |
| fixed_nll | 0.016099 |
| fixtures | 0.711806 |
| likelihood_preparation | 6.105605 |

Normalization resolves to `Hermitian` for both model sizes. Reference twelve-wave diagnostics retain 1,248 bytes of normalization statistics and 4,000,096 bytes for the observed prepared dataset. These are existing runtime counters; they do not cover every allocation or represent an aggregate cross-section heap count.

Parquet source accounting includes fixture writes, decoded row loading, and subsequent mass-binning traversals. Loading uses a warm filesystem cache.

**Scientific gate: passed the roundoff-aware reproducibility check.** All one-thread repeat groups and both Parquet thread groups have identical scientific output. Four-thread likelihood and bootstrap groups show tiny repeat drift. Largest observed same-configuration absolute spread: 1.8189894035458565e-12. The gradient path uses parallel folds and merges (`crates/laddu-runtime/src/cpu/reduction.rs:715`), and the observations are consistent with floating-point reduction grouping. These differences are accepted as roundoff under the maintainer's clarified policy; no change to library reductions is required.

`scientific-gate.json` preserves the earlier failed bitwise self-comparison. `roundoff-gate.json` records the passing self-comparison under the revised ADR 0024; issue 14 is closed as expected parallel roundoff. Comparison uses four times baseline spread across repeats/thread counts, with a 32-ULP floor. A separate repeat-spread guard (`1e-12 * max(1, |values|)`) rejects materially unstable calibration; candidate measurements never enlarge the field-specific envelope. Raw measurements, scientific outputs, and their original source/binary hashes remain unchanged.

The full scientific values, gradients, fitted parameters, replica fits, paired projected draws, bootstrap uncertainties, cache diagnostics, stage records, execution settings, binary/source hashes, and machine/build metadata are preserved in `baseline.json`. `summary.json` provides compact routine results.

## Replica scaling

The separate reference run fitted 200 real paired bootstrap replicas using four threads. Status: `ok`; total 79.716 seconds; peak RSS 986.10 MB. Preserved in `stress.json`.

This single stress measurement checks replica scaling and the memory bound; it cannot establish repeat determinism or timing variation.

## Controlled memory-stop check

A quick-mode run with a deliberately reduced 1,000,000-byte limit stopped all four workflows as `memory_limit`, preserved exit status/stderr and any completed stage records, and returned exit code 1. The ordinary reference baseline did not fail at 28 GB. This lower-limit run tests the stop mechanism; it does not represent a reference memory failure. See `memory-limit.json`.
