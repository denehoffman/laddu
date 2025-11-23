# Agents Guide

## Tooling Defaults
- Python lives under `py-laddu`/`py-laddu-mpi`; use the `mise` tasks (which wrap `uv`, Cargo, and `maturin`) for environment management and commands so you rarely need to call the tools directly.
- Regenerate Python lockfiles by running `uv lock` in each package directory (`py-laddu`, `py-laddu-mpi`) whenever their dependencies change.
- Run `ruff` for linting/formatting Python code; prefer `mise run lint-python` (or `uv run ruff ...`) so it shares the same virtual environment that `mise run develop` maintains.
- For Rust crates, make `cargo clippy --all-targets --all-features` part of your pre-commit check, alongside the existing test targets below.

## Documentation Expectations
- Python docstrings are primarily authored in the Rust bindings (PyO3 modules). Write them in [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html), including `Parameters`, `Returns`, and `Examples` sections. Keep examples runnable snippets that show real Python usage.
- Rust code must document every public item (`pub` structs, enums, traits, functions, modules). Provide enough context for users to understand invariants and edge cases, and prefer doctests (`/// ```rust
/// let ...
/// ``` `) whenever an example can compile.
- Align the Python and Rust docs: whenever a Rust function backs a Python surface API, ensure the docstring explains the same behavior and, even though cross-linking between Python and Rust docs is usually impossible, call out where the canonical Rust docs live.

## Handy `mise` Commands
Use `mise run <task>` from the repo root; each task wires up `uv`, Cargo, and other tooling for you.

| Command | Purpose |
| --- | --- |
| `mise run develop` / `mise run develop-tests` | Build/install the CPU wrapper (optionally with tests extras) via `maturin` + `uv`. |
| `mise run develop-mpi` / `mise run develop-mpi-tests` | Build/install both CPU and MPI wrappers, wiring in extras as needed. |
| `mise run test` / `mise run test-mpi` | Run combined Rust + Python tests (standard or MPI features). |
| `mise run docs` | Build Sphinx docs alongside `cargo doc`. |
| `mise run lint` / `mise run lint-mpi` | Run `cargo clippy` plus Python Ruff checks (standard or MPI). |
| `mise run docker-build` / `mise run docker-shell` | Build the dev container or drop into the synced shell inside it. |
| `mise run coverage` | Generate both Rust (llvm-cov) and Python (pytest --cov) coverage reports once dev artifacts exist. |

## Code Coverage
- Always run `mise run develop-tests` first so the shared `.venv` contains the pytest extras (including `pytest-cov`) and the Rust/Python artifacts are consistent.
- **Rust**: run `cargo llvm-cov --workspace --lcov --output-path coverage-rust.lcov --summary-only --exclude-from-report py-laddu -F rayon` from the repo root. This mirrors `.github/workflows/coverage.yml` (nightly toolchain + rayon feature) and emits an LCOV file that Codecov ingests.
- **Python**: with the `.venv` activated (or by calling the interpreter directly), execute `.venv/bin/python -m pytest --cov --cov-report xml:coverage-python.xml` from the repo root. `pytest.toml` already points at `py-laddu/tests` and the package modules, so no extra flags are needed.
- We currently treat MPI-heavy paths and PyO3 shims inside `laddu-python` as out-of-scope for unit coverage because MPI coordination is difficult to exercise locally and the Python tests already hit the exposed bindings. The Python coverage config omits `laddu/mpi.py`, and the Rust MPI module is wrapped in `#[cfg_attr(coverage_nightly, coverage(off))]` so `cargo llvm-cov` won't report it.
- When adding tests, aim for parity between Rust and Python suites: if logic exists on both sides (e.g., bindings that mirror a Rust helper), keep the scenarios aligned. It's fine for a test to live in just one language when the behavior is language-specific, but default to keeping feature coverage in sync.

## Working Style Checklist
1. Run the appropriate `mise run develop*` task (add the `-tests`/MPI variants as needed) before touching Python code so the managed `uv` environment lines up.
2. Keep `ruff` and `cargo clippy` clean alongside the standard Cargo/Python tests (`mise run test`). Use `mise run lint-python` / `mise run lint-rust` to make sure code is clear of lint/type issues.
3. When adding APIs, write Rust docs + doctests first, then mirror concise NumPy-style docstrings (with usage examples) on the Python side.
4. Prefer extending `mise` tasks rather than adding ad-hoc scripts if you need a repeatable workflow—keep the table above in sync as new tasks appear.

## Git Workflow
- Follow Conventional Commits for every commit message (e.g., `feat: ...`, `fix: ...`).
- Never commit directly to `main`; if an integration branch already exists for the effort, use it.
- When no appropriate branch is available, ask whether to create a new branch or reuse an existing one before proceeding.
