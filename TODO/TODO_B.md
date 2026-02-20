# TODO Phase B: Early Architecture Decision Spikes (Timeboxed)

Scope: `O074`, `O078`, `O079`, `O081`, `O075`, `O080` + optional `O082` + gates `G1`-`G4`

## Goals
- Run high-impact prototypes early enough to avoid wasted downstream implementation.
- Decide architecture with explicit go/no-go criteria.


Subtask tags: `[Phase.Objective.Step][E:S|M|L][D:<dependency step id|none>]`.
## Timeboxing Guidance
- Each spike: 3-7 days.
- Output required: prototype code, benchmark delta, migration impact note.

## O074: Execution Context Prototype
### Subtasks
- [x] [B.O074.01][E:S][D:none] Define `ExecutionContext` API owning thread pool + scratch allocators.
- [x] [B.O074.02][E:M][D:B.O074.01] Integrate into one hot path (`evaluate` + `evaluate_gradient`) behind feature flag.
- [x] [B.O074.03][E:S][D:B.O074.02] Benchmark setup overhead and repeated-call performance.
- [x] [B.O074.04][E:S][D:B.O074.03] Document lifecycle (create, reuse, drop) and thread-policy interactions.
  - Completed 2026-02-17: `e6ac649`

### Pseudocode
```rust
let ctx = ExecutionContext::new(ThreadPolicy::Global);
for p in params_iter {
    let out = evaluator.evaluate_with_ctx(p, &ctx)?;
}
```

### Gate G1 Decision Inputs
- Setup overhead reduction.
- Simplicity of API integration.
- Memory footprint change.

## O078: SoA Dataset Core Prototype
### Subtasks
- [x] [B.O078.01][E:M][D:B.O074.04] Build minimal SoA storage for p4/aux/weight.
  - Completed 2026-02-17: `777d29b`
- [x] [B.O078.02][E:M][D:B.O078.01] Provide row-view adapter compatible with current variable/evaluator APIs.
  - Completed 2026-02-17: `d83ac25`
- [x] [B.O078.03][E:M][D:B.O078.02] Define SoA-native event access interfaces for amplitudes/variables.
  - Completed 2026-02-17: `92d3b2f`
- [x] [B.O078.04][E:M][D:B.O078.03] Add SoA-native amplitude/variable traits and evaluator plumbing.
  - Completed 2026-02-17: `7d02228`
- [x] [B.O078.05][E:M][D:B.O078.04] Implement SoA-native storage views consumed directly by evaluator kernels (no row materialization).
  - Completed 2026-02-17: `2218067`
- [x] [B.O078.06][E:M][D:B.O078.05] Pivot contract: keep legacy AoS evaluate path and add cache-centric evaluate path (`evaluate_cached`/`evaluate_gradient_cached`) for direct benchmark comparison.
  - Completed 2026-02-17: `4263812`
- [x] [B.O078.07][E:M][D:B.O078.06] Add SoA-native `precompute_with_access` path and keep AoS precompute path for A/B benchmark comparison.
  - Completed 2026-02-17: `f067b43`
- [x] [B.O078.08][E:M][D:B.O078.07] Refactor built-in amplitudes so compute depends on parameters + cache only; move event reads to precompute stage.
  - Completed 2026-02-17: `412724e`
- [x] [B.O078.09][E:M][D:B.O078.08] Refactor variable-dependent amplitude inputs into precomputed event bases consumed by cache-centric evaluate methods.
  - Completed 2026-02-17: `5716427`
- [x] [B.O078.10][E:S][D:B.O078.09] Add parity tests and explicit assertions that cached evaluate paths do not iterate dataset events.
  - Completed 2026-02-17: `416f6bb`
- [x] [B.O078.11][E:S][D:B.O078.10] Benchmark AoS-vs-cached evaluate and AoS-precompute-vs-SoA-precompute throughput.
  - Completed 2026-02-17: `3f23c2d`
- [x] [B.O078.12][E:S][D:B.O078.11] Profile cached evaluate hot paths (`evaluate_cached_local` and `evaluate_gradient_cached_local`) and identify dominant overhead contributors relative to AoS paths.
- [x] [B.O078.13][E:M][D:B.O078.12] Reduce cached evaluate overhead in hot loops (dispatch/error-handling fast paths where safe) and re-measure AoS-vs-cached value/gradient performance.
- [x] [B.O078.14][E:M][D:B.O078.13] Expand analytic/cache-native `compute_gradient_cached` coverage for amplitudes still relying on generic finite-difference fallback and quantify impact.
- [x] [B.O078.15][E:M][D:B.O078.14] Optimize `load_soa` precompute throughput by eliminating avoidable conversion/allocation overhead and re-benchmark AoS-vs-SoA load performance.
- [x] [B.O078.16][E:S][D:B.O078.15] Add stage-isolated microbenchmarks for cached value fill, cached gradient fill, expression eval, and precompute/load components.
- [x] [B.O078.17][E:S][D:B.O078.16] Re-run CPU benchmark suite and document pass/fail thresholds plus observed deltas for AoS-vs-cached and AoS-vs-SoA paths.
- [x] [B.O078.18][E:S][D:B.O078.17] Isolate `load` vs `load_soa` cost components (dataset conversion/setup/precompute) and capture baseline timings with profiling runs driven by a dedicated `bin/main.rs` executable (not Criterion benchmarks).
- [x] [B.O078.19][E:M][D:B.O078.18] Add per-amplitude AoS-vs-SoA precompute timing instrumentation in the dedicated profiling executable and rank dominant SoA hotspots.
  - Completed 2026-02-18: `a591ab7`
- [x] [B.O078.20][E:M][D:B.O078.19] Implement targeted SoA precompute/load optimizations for top-ranked hotspots and preserve AoS parity checks.
  - Completed 2026-02-18: `a591ab7`
- [x] [B.O078.21][E:S][D:B.O078.20] Re-run CPU benchmark/report workflow plus dedicated profiling executable and document updated AoS-vs-SoA load/precompute deltas.
  - Completed 2026-02-18: `a591ab7`
- [x] [B.O078.22][E:S][D:B.O078.21] Write acceptance note for SoA load/precompute performance: match/exceed AoS when feasible, or justify residual slowdown as acceptable one-time cost.
  - Completed 2026-02-18: `a591ab7`
  - Accepted 2026-02-18: latest CPU benchmark summary (`target/criterion/summary/benchmark_summary_cpu.json`) shows cached/local and SoA precompute paths ahead of AoS counterparts: `evaluate_cached_local` vs `evaluate_aos_local` -0.7408% time, `evaluate_gradient_cached_local` vs `evaluate_gradient_aos_local` -0.8067%, `load_soa_precompute` vs `load_aos_precompute` -0.8608%, and `soa_precompute_only` vs `aos_precompute_only` -1.1592%.

### Gate Considerations
- Adapter complexity.
- Measured CPU gains in realistic workloads.

## O079: Arrow-Native Interchange Prototype
### Subtasks
- [x] [B.O079.01][E:M][D:B.O078.04] Prototype direct Arrow ingestion + export path (Rust + Python boundary).
  - Completed 2026-02-18: `a591ab7`
- [x] [B.O079.02][E:M][D:B.O079.01] Avoid intermediate list/object materialization for large arrays.
  - Completed 2026-02-18: `91249f6`
- [x] [B.O079.03][E:S][D:B.O079.02] Compare copy counts and peak RSS against current pathways.
  - Completed 2026-02-18: `466f48d`
- [x] [B.O079.04][E:S][D:B.O079.03] Validate compatibility with parquet/root/numpy/pandas/polars entrypoints.
  - Completed 2026-02-18: `829c617`

### Gate G2 Decision Inputs
- End-to-end copy reduction.
- Integration complexity with existing Dataset model.

### Gate G2 Readiness Assessment (2026-02-18)
- `SoA-ready`:
  - Parquet open path: direct `read_parquet_soa` is implemented and benchmarked faster than AoS in current CPU benchmarks.
  - ROOT open path: direct `read_root_soa` is implemented and benchmarked at parity/slight edge versus AoS in current CPU benchmarks.
  - SoA evaluator/precompute path: explicit `load_soa` and SoA-native precompute/compute paths exist and are benchmark-covered.
- `Needs SoA API promotion`:
  - Rust public APIs are still primarily `Dataset`-first in high-level usage patterns.
  - Python surface is still `Dataset`-centric; no first-class `DatasetSoA` Python type/entrypoints yet.
- `Needs compatibility shim or migration step`:
  - Row-centric transforms/helpers (`EventData`-based workflows) remain the default for many utility paths.
  - IO write surface is mixed (`write_parquet_soa` exists; no direct `write_root_soa` entrypoint yet).
  - Existing examples/tests/downstream code assume AoS object semantics.

### Proposed SoA Migration Stages
1. `Stage 1 (default switch with compatibility)`:
   - Keep AoS types available, but route default load/evaluate hot paths through SoA-backed implementations.
   - Add compatibility adapters only at boundaries that still require row events.
2. `Stage 2 (API promotion)`:
   - Add first-class SoA-facing APIs in Rust/Python and migrate examples/tests to SoA-first usage.
   - Restrict new features to SoA path to avoid dual-maintenance growth.
3. `Stage 3 (AoS deprecation/removal)`:
   - Deprecate AoS-first entrypoints after one compatibility window.
   - Remove AoS internals once benchmark parity/superiority and migration completeness are confirmed.

### G2 Decision (Current)
- Decision: `Adopt now (staged)`
- Rationale: current benchmark and profiling evidence supports SoA for open/precompute/evaluate hot paths; remaining work is API migration and compatibility management rather than core performance feasibility.
- Key risk: temporary dual-path maintenance during staged rollout.

## O081: SoA-First And Cache-Only Full Adoption
### Subtasks
- [x] [B.O081.01][E:M][D:B.O079.04] Define migration contract: `Dataset` becomes SoA-backed canonical storage and cached evaluate paths become canonical `evaluate`/`evaluate_gradient`.
  - Completed 2026-02-18: `d9abbed`
  - Additional local progress 2026-02-19: `1940db2` (cache-only `compute`/`compute_gradient` API adoption across core/amplitudes/extensions, canonical evaluate naming cleanup, and `EvaluatorSoA` MPI evaluate/evaluate_gradient parity restoration).
  - Additional local progress 2026-02-19: `3189074` (removed storage-specific evaluator API surface, routed canonical dataset load/write paths through columnar internals, and migrated public trait/callsite signatures to canonical `Dataset`-first interfaces).
- [x] [B.O081.02][E:M][D:B.O081.01] Rename core data types so public `Dataset` API maps to SoA storage internals; isolate/remove legacy row-storage internals behind temporary internal adapters only where required.
  - Additional local progress 2026-02-19: `a1a2439` (internalized `Dataset.events`, added canonical `events_local` accessor usage across core/extensions/python, and removed direct row-storage field access from dependent crates).
- [x] [B.O081.03][E:M][D:B.O081.02] Promote SoA IO as canonical (`read_parquet`, `read_root`, write/read roundtrips) and remove AoS-first loading defaults across core APIs.
  - Additional local progress 2026-02-19: `1e6290f` (removed dead AoS-first parquet loader/conversion helpers and made canonical `read_*`/`write_parquet` paths directly route through columnar-backed dataset internals).
- [x] [B.O081.04][E:M][D:B.O081.03] Remove legacy non-cached evaluator paths; rename cached methods to canonical names and update evaluator interfaces accordingly.
  - Additional local progress 2026-02-19: `c40d444` (removed redundant self-comparison evaluator tests/benchmarks and canonicalized remaining benchmark labels to reflect single-path cache-only evaluation semantics).
- [x] [B.O081.05][E:M][D:B.O081.04] Update amplitude traits and implementations to SoA/cache-only execution model; remove obsolete compute/evaluate entrypoints.
  - Additional local progress 2026-02-19: `9c8c4da` (removed dual precompute entrypoints, unified canonical names to `precompute`/`precompute_all` on view-based execution, and migrated all built-in amplitudes + profiling/benchmark callsites).
- [x] [B.O081.06][E:M][D:B.O081.05] Update variable trait and built-in variable implementations to SoA-first `Dataset`/event-view access and remove AoS-only callpaths.
  - Additional local progress 2026-02-19: `6c2303c` (removed `EventAccess`/`NamedEventAccess`, switched `Variable` evaluation to canonical `NamedEventView`, removed AoS-only variable codepaths, and made indexed event-view accessors panic on OOB with direct-return APIs).
- [x] [B.O081.07][E:M][D:B.O081.06] Migrate `laddu`, `laddu-amplitudes`, and `laddu-extensions` callsites/examples to new canonical `Dataset` and evaluator APIs.
  - Additional local progress 2026-02-19: `3dcc4e4` (updated `profile_kmatrix` example to handle canonical evaluator `Result` return values explicitly in hot-loop profiling modes).
  - Additional local progress 2026-02-19: `47050e8` (canonicalized `profile_cached_paths` mode names/output labels and synced `benches/Justfile` + `benches/README.md` file-open/profiling commands to canonical naming with backward-compatible aliases).
  - Additional local progress 2026-02-19: `6551f54` (removed retained legacy profiling mode aliases and legacy Justfile alias tasks so only canonical profiling command names remain).
- [x] [B.O081.08][E:M][D:B.O081.07] Migrate Python bindings and `py-laddu` API surface to new canonical semantics while preserving user-facing class/function names (`Dataset`, `evaluate`, etc.).
  - Additional local progress 2026-02-19: `5cfdb9d` (audited Python bindings/stubs for transitional API names and removed remaining "legacy" naming in dataset metadata precedence test fixtures).
  - Additional local progress 2026-02-19: `d61f6ee` (completed Python-facing docstring pass and corrected residual wording in Dataset binning return docs to canonical neutral phrasing).
- [x] [B.O081.09][E:S][D:B.O081.08] Remove deprecated AoS compatibility shims and dead code paths; ensure no remaining production references to legacy AoS evaluator/dataset internals.
  - 2026-02-19: Legacy AoS-equivalence dataset tests removed; canonical storage IO parity/roundtrip tests retained (`28f98b5`).
  - Additional local progress 2026-02-19: `01915ee` (removed remaining transitional migration-note references from canonical `Dataset`/`Evaluator` public docs and top-level crate docs).
- [x] [B.O081.10][E:S][D:B.O081.09] Rebaseline benchmarks: CPU Criterion (including file-open group), profiling binary modes, and Python benchmark smoke checks under SoA-first/cached-only APIs.
  - Completed 2026-02-19: `9434180` (CPU Criterion + JSON report rebaseline rerun, profiling `parquet_open`/`root_open`, Python smoke via `just test-python`, and benchmark README baseline refresh).
- [x] [B.O081.11][E:S][D:B.O081.10] Execute full validation matrix (`cargo test`, targeted crate tests, `just test-python`, `prek run --all-files`) and document migration breakage notes.
  - Additional local progress 2026-02-19: validation matrix rerun passed (`cargo test`, `just test-python`, `prek run --all-files`); migration notes are intentionally not recorded in `CHANGELOG.md` because changelog is release-generated.
- [x] [B.O081.12][E:S][D:B.O081.11] Final adoption gate: confirm SoA-first/cached-only architecture readiness and record follow-up cleanup tasks for post-adoption optimization.
  - Additional local progress 2026-02-19: adoption-gate audit found no remaining legacy non-cached evaluator production APIs; one residual legacy wording string in profiling binary was removed.
  - Completed 2026-02-19: `e1180ba` (removed residual legacy wording from profiling binary and finalized O081 adoption gate sweep).
  - Follow-up cleanup candidates (post-adoption optimization):
    - Rename remaining internal `columnar`/`DatasetStorage` identifiers to canonical dataset naming.
    - Reduce transitional `EventData` conversion surfaces that still exist for constructor/testing compatibility.
    - Keep profiling/benchmark wording aligned with canonical semantics to avoid drift.
- [x] [B.O081.13][E:M][D:B.O081.12] Define canonical Python-reader ingestion contract around Arrow (`RecordBatch`/`Table`) and implement a shared Rust entrypoint for Arrow-to-`Dataset` SoA loading.
  - Completed 2026-02-19: `097eadd` (centralized dataset IO inference/helpers in `laddu-core::data::io`, removed duplicated Python-side inference paths, made p4 suffix handling case-insensitive across interfaces, and tightened Python ingestion typing).
- [x] [B.O081.14][E:M][D:B.O081.13] Migrate `from_pandas` and `from_polars` to the shared Arrow ingestion path with borrow-first / zero-copy-fast-path behavior.
  - Completed 2026-02-19: `097eadd` (routed pandas/polars ingestion through shared Arrow-backed backend path with optional direct `pyarrow` imports and concrete type hints).
- [x] [B.O081.15][E:M][D:B.O081.14] Migrate `from_dict` to Arrow-builder ingestion with typed fast paths and explicit fallback handling for incompatible layouts.
  - Completed 2026-02-19: `e309084` (added typed column normalization + Arrow-builder fast path for `from_dict`, with explicit fallback handling for unavailable/incompatible Arrow layouts).
- [x] [B.O081.16][E:M][D:B.O081.15] Migrate Uproot/AmpTools ROOT Python reader paths to the shared Arrow ingestion pathway and normalize schema mapping.
  - Completed 2026-02-19: `70f412e` (routed Uproot/AmpTools reads through shared normalized `from_dict` ingestion and replaced bespoke AmpTools event materialization with canonical column mapping).
- [x] [B.O081.17][E:S][D:B.O081.16] Add ingestion benchmarks (time + peak RSS + copy diagnostics) across Python readers and compare against pre-migration baselines.
  - Completed 2026-02-19: `9937b85` (added Python ingestion benchmark harness with time/peak RSS/copy diagnostics and optimized polars fallback conversion to use direct NumPy column extraction).
- [x] [B.O081.18][E:S][D:B.O081.17] Finalize reader migration cleanup: remove superseded ingestion plumbing, lock in canonical docs/tests, and record residual non-zero-copy cases.
  - Completed 2026-02-19: `e47af80` (added ingestion regression test for non-contiguous inputs and benchmark README documenting canonical execution and residual non-zero-copy pathways).

### Adoption Gate Inputs
- API consistency after renaming (`Dataset`, `evaluate`, `evaluate_gradient`).
- Absence of legacy AoS/non-cached production paths.
- Benchmark parity/superiority and validation stability across Rust + Python surfaces.

### O081.01 Contract Details (Approved Direction)
- Canonical dataset contract:
  - Public `Dataset` name will represent SoA-backed storage.
  - Existing row-oriented storage (`EventData` collections) is transitional and slated for removal from production pathways.
- Canonical evaluation contract:
  - Public `evaluate` and `evaluate_gradient` will be cache-only paths.
  - Legacy non-cached evaluate pathways are transitional and slated for removal.
- API-name preservation requirement:
  - Keep user-facing names (`Dataset`, `evaluate`, `evaluate_gradient`) while replacing internals.
  - Prefer migration via internal renames/adapters during intermediate commits, then remove adapters once parity is validated.
- Cross-crate migration map:
  - `laddu-core`: dataset storage internals, IO defaults, evaluator/amplitude/variable traits.
  - `laddu`, `laddu-amplitudes`, `laddu-extensions`: callsite migration to canonical SoA/cache-only APIs.
  - `laddu-python` and `py-laddu`: preserve public Python names while migrating backend semantics.

## O075: Expression IR Prototype
### Subtasks
- [x] [B.O075.01][E:M][D:B.O081.18] Define IR nodes + pass pipeline (CSE, constant fold, activation specialization).
- [ ] [B.O075.02][E:M][D:B.O075.01] Compile from existing expression tree into IR once per load.
- [ ] [B.O075.03][E:M][D:B.O075.02] Execute IR for value/gradient on representative models.
- [ ] [B.O075.04][E:S][D:B.O075.03] Compare with current evaluator for speed and maintainability.

### Pseudocode
```rust
let ir = compile_expression(expr)
    .run_pass(Pass::CSE)
    .run_pass(Pass::ConstantFold)
    .run_pass(Pass::ActivationSpecialize(mask));
```

### Gate G3 Decision Inputs
- Speedup in hot paths.
- Correctness parity difficulty.
- Maintenance cost of dual backends.

## O080: MPI Objective Model Prototype
### Subtasks
- [ ] [B.O080.01][E:L][D:B.O075.04] Prototype rank-local event ownership objective.
- [ ] [B.O080.02][E:M][D:B.O080.01] Reduce only aggregate objective/gradient values by default.
- [ ] [B.O080.03][E:M][D:B.O080.02] Keep explicit gather path for per-event outputs.
- [ ] [B.O080.04][E:S][D:B.O080.03] Measure communication reduction and semantic/API impact.

### Gate G4 Decision Inputs
- Collective traffic reduction.
- User-facing semantic changes and migration cost.

## O082: Optional Ganesh Switch-Over Batch (Execution Context + Threads)
### Subtasks
- [ ] [B.O082.01][E:M][D:B.O080.04] Optional Ganesh update: finalize shared `ExecutionContext` and explicit thread policy modes (`Single`, `GlobalPool`, `Dedicated(n)`).
- [ ] [B.O082.02][E:M][D:B.O082.01] Plumb Laddu-to-Ganesh context/thread-policy forwarding so one runtime policy controls both crates.
- [ ] [B.O082.03][E:S][D:B.O082.02] Add cross-crate microbenchmarks comparing per-call setup vs reused context through wrappers.

## Gate Outputs Template (for each gate)
- Decision: `Adopt now` / `Defer` / `Drop`
- Rationale:
- Benchmark summary:
- Migration impact:
- Follow-up actions:

## Recommended Execution Order (Within Phase)
1. `O074`
2. `O078`
3. `O079`
4. `O081`
5. `O075`
6. `O080`
7. `O082` (optional Ganesh switch-over batch)
