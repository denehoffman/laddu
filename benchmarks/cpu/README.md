# Generated CPU workflow baseline

These local `laddu` workloads establish a baseline before CPU optimization. Timing
fluctuations do not gate ordinary CI. The runner needs Linux `/proc`, Python 3.11+
with only its standard library, and the repository's Rust toolchain.

Run the complete baseline, including repeated four-thread and one-thread results:

```sh
python3 scripts/cpu_benchmark.py run --variation --output target/cpu-benchmark/baseline.json.gz
```

The runner builds `cpu_workflow` in release mode before measurement. Each case
uses a fresh process; cases run sequentially. Run on an otherwise idle machine.
`--no-build` reuses an already-built worker; the result records its SHA-256,
compiler, revision, working tree, machine, and build settings. Set `--binary` if
using a different Cargo target directory.

For a manual build used with `--no-build`, match the runner's feature flags:

```sh
cargo build -p laddu --example cpu_workflow --release --no-default-features --features fit
```

Inspect the frozen inputs without running anything:

```sh
python3 scripts/cpu_benchmark.py run --plan --variation
```

| Size | Data events | Accepted/generated MC events | Parquet events per period |
| --- | ---: | ---: | ---: |
| quick | 128 | 512 | 512 |
| small | 2,000 | 20,000 | 20,000 |
| reference | 20,000 | 200,000 | 200,000 |

The reference size is frozen for this effort. Four related cases cover three
workflows without a Cartesian product:

- Five- and twelve-wave coherent models: fixture construction, likelihood
  preparation, ten value evaluations, ten value-plus-gradient evaluations, and
  an L-BFGS-B fit bounded by both parameter limits and 30 steps.
- Twelve-wave cross sections: the same preparation and central fit, eight actual
  resampled likelihoods fitted through `Ensemble::bootstrap_fit`, scalar total
  construction, and two batched projections with all twelve tagged components.
- Four-period Parquet: fixture writing, metadata opening, actual row loading,
  and `Dataset.bin_by` mass partitioning. Each period has four shards; each mass
  partition remains alive until its statistics are collected.

Models use complex Fourier waves modulated by mass and angle. The first wave's
imaginary coupling is fixed to remove the overall phase ambiguity. Data and MC
are independently generated uniform scalar samples: mass in [1, 2), cos(theta)
in [-1, 1), phi in [-pi, pi), and positive weights uniform in [0.75, 1.25).
They exercise weighted fitting and derived-measurement contracts; they are not
a physics closure study. Seed 20261002 and all role/shard/replica offsets live in
the worker. Bootstrap fits start at the central fitted parameters and retain
their likelihood replicas, parameter draws, replica order, and seed.

Execution uses CPU f64, fixed four threads, disabled JIT, automatic normalization,
and the `fastest` dataset policy. The JSON records the resolved normalization
strategy, resident cache bytes, retained normalization bytes, per-replica
diagnostics, process/resource memory reports, stage and total times, source
traversals and rows, fit outputs, cross sections, bin values, paired draws, and
bootstrap standard deviations. Pool/resource reservations are the existing
runtime counters, not an estimate of all heap allocations. Replica diagnostics
can describe shared allocations and must not be summed as unique bytes.

Peak RSS comes from the kernel's lifetime `VmHWM`, sampled by the parent during
work and read by the worker at each completed stage and exit. Resource-report
sampling and scientific result materialization contribute to total workflow
time. `process_seconds` also includes startup and output serialization; builds
are excluded. Parquet generation is timed separately; loading follows writing,
so this measures warm filesystem-cache reads.

For a short correctness check, use:

```sh
python3 scripts/cpu_benchmark.py run --sizes quick --repeats 1
LADDU_CPU_WORKER="$PWD/target/release/examples/cpu_workflow" .venv/bin/python -m pytest -q scripts/tests/test_cpu_benchmark.py
```

Quick mode uses three evaluations and four fit steps. It does not establish a
timing baseline. The default command-line tests avoid native workloads unless
`LADDU_CPU_WORKER` is set; they impose no timing thresholds.

Run the distinct replica-scaling case when changing bootstrap memory behavior:

```sh
python3 scripts/cpu_benchmark.py run --sizes reference --stress --repeats 1 --output target/cpu-benchmark/stress.json.gz
```

This uses 200 fitted, paired bootstrap replicas, including the full component
projection path. It does not replace the routine eight-replica measurements.

For standalone scalar preparation, select `--workflows bootstrap-scalar`.
It uses the same native fitted bootstrap replicas, but constructs the total
before evaluating projections. `--workflows bootstrap-arbitrary` exercises the
same standalone API with independently generated observed and accepted rows per
replica. Each replica is fitted from the central parameters; its seeds are
`seed + 1000 + index` for data and `seed + 2000 + index` for accepted MC.
This arbitrary ensemble has no native bootstrap sharing provenance or bootstrap
seed. It tests the conservative fallback, rather than representing a resampling
study. Event counts, weights, model, fit limits and generated MC stay identical.
`bootstrap-arbitrary-data` keeps the central accepted MC but gives each fitted
replica independently generated observed rows. Its ensemble also has no native
sharing provenance: standalone scalar evaluation conservatively prepares each
replica, while projections may share evaluation using the accepted dataset's
identity. This mode isolates temporary integral-cache scaling from repeated
accepted-normalization preparation and component-plan compilation.
These opt-in modes work with both routine eight-replica and `--stress` runs:

```sh
python3 scripts/cpu_benchmark.py run --workflows bootstrap-scalar,bootstrap-arbitrary --sizes reference --variation
python3 scripts/cpu_benchmark.py run --workflows bootstrap-arbitrary --sizes reference --stress --variation
python3 scripts/cpu_benchmark.py run --workflows bootstrap-arbitrary-data --sizes reference --stress --variation
```

Stage records include live reservations and their high-water mark; final
likelihood diagnostics describe retained data caches and normalization statistics.
Do not sum shared native replica diagnostics as unique allocations.

The RSS watchdog stops a worker above 28,000,000,000 bytes by default, saves
completed stage records, and marks the case `memory_limit`. Its 10 ms polling
interval permits overshoot; it is a controlled benchmark stop rather than a
kernel-enforced allocation cap. Abnormal termination without a measured limit
breach is reported as `failed`, with the exit status and stderr preserved.
Use a lower limit to exercise this path:

```sh
python3 scripts/cpu_benchmark.py run --sizes quick --repeats 1 --memory-limit-gb 0.001 --output target/cpu-benchmark/limited.json
```

If a reference baseline exceeds 28 GB, keep its failure artifact and compare
the old and new versions using identical `--sizes small` commands as well.

Compare before and after artifacts from the same machine, compiler, profile,
fixture, thread settings, and counts:

```sh
python3 scripts/cpu_benchmark.py compare target/cpu-benchmark/baseline.json.gz target/cpu-benchmark/after.json.gz
```

The comparison reports median stage-time, total-time, and RSS ratios. Scientific
comparison includes every fitted parameter, likelihood, replica draw, cross
section, projected value, bootstrap uncertainty, bin count, and weight sum.
For each numeric field the absolute tolerance is the larger of four times its
observed baseline spread across repeats/thread counts and 1,024 ULPs at the largest
baseline magnitude. The ULP floor allows small reduction changes when the
measured spread is zero. Baseline variation must be collected before optimizing;
candidate outputs never widen tolerances. Same-thread repeats may differ by
floating-point roundoff: their spread must stay below `1e-12 * max(1, |values|)`.
This guard rejects materially unstable baselines before calibration can turn
them into broad tolerances. Candidate results must also stay within the tighter
field-specific baseline-calibrated envelope; this roundoff guard never widens it.
A missing/failed case, changed output
shape, nonfinite result, or scientific difference beyond the tolerance fails
comparison. Performance ratios themselves have no pass/fail threshold.

On 2026-10-03 the maintainer approved increasing the ULP floor from 32 to 1,024
after the unchanged-worker control reached 548 ULPs of variation. The new floor
allows at most about `2.3e-13` relative error for normal nonzero f64 values. The
baseline-spread multiplier and `1e-12` repeat-instability guard remain unchanged.
Comparison reports record the ULP floor; preserve older strict reports when
rechecking existing observations under the revised policy.

The checked-in result artifacts and `baseline.md` document this checkout's
baseline, stress run, and memory-stop check. Keep those artifacts unchanged;
write later measurements under `target/` and present before/after results with
each optimization ticket.

The raw results retain the original runner hash and observations. The initial
`scientific-gate.json` records the superseded bitwise-equality check;
`roundoff-gate.json` records acceptance under the current scientific-reproducibility
policy. Reinterpreting existing measurements does not change their timing or
scientific outputs.

For performance ticket 03, the worker uses shared total/projection construction.
Its `cross_section_total` stage now includes evaluation of the requested
projections, while `component_projections` retrieves their retained outputs.
Compare the **sum of these two stages** with the original worker's separate
construction and projection stages, as well as total workflow time and peak RSS.
The frozen model, inputs, fits, scientific outputs, and runner are unchanged.

Set `LADDU_CPU_VERIFY_SHARED=1` for a separate differential correctness run.
The worker also evaluates the standalone construction/projection APIs using the
same fitted parameters and paired replica objects and records their scientific
outputs in `diagnostics.separate_science`. Compare them using the tolerances
from the original baseline. These verification runs execute both paths and
must not be used as performance measurements.

Print a compact summary of a JSON or compressed JSON artifact:

```sh
python3 scripts/cpu_benchmark.py summary target/cpu-benchmark/baseline.json.gz
```

For projection evaluation, sweep component, paired replica, and projection counts:

```sh
python3 scripts/cpu_projection_study.py --binary target/release/examples/cpu_workflow --output target/cpu-benchmark/projection-study.json.gz --variation --stress
```

The twelve-wave study varies component count (0, 1, 6, 12), paired replica count
(2, 8, and optionally 200), and projection count (1, 2, 4), one dimension at a
time. Four projections include finer mass bins and a joint mass/angle projection.
Default `cpu_benchmark.py` cases retain their frozen configuration.

Use `--size small --batch-profile --repeats 1` for a separate diagnostic that
times cache construction and parameter evaluation on the first accepted batch
(at most 8,192 events), individually for the full model and each component.
Component-plan preparation and bin accumulation are outside these timers.
Projection sets already deduplicate axes and canonical tags and read each source
once; projection count changes accumulation work, rather than multiplying event
evaluation. Diagnostic runs add work and must be excluded from workflow and RSS
comparisons.

Use `--verify-blocks` for a separate correctness diagnostic. At identical fitted
central parameters and real paired bootstrap draws, it compares parallel block
outputs with the existing one-thread serial visitor, checking both components
of each complex value by their bit patterns. It retains reference outputs for
one component at a time and records the verified count. Exclude these runs from
performance and memory comparisons.
