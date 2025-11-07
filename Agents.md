# Agents Guide

## Tooling Defaults
- Python lives under `py-laddu`/`py-laddu-mpi`; always use `uv` for environment management, installation, and running commands (see the `.justfile` recipes that wrap `uv pip`, `uv run`, and `maturin`).
- Run `ruff` for linting/formatting Python code; prefer `uv run ruff ...` so it shares the same virtual environment that `just develop*` sets up.
- For Rust crates, make `cargo clippy --all-targets --all-features` part of your pre-commit check, alongside the existing test targets below.

## Documentation Expectations
- Python docstrings are primarily authored in the Rust bindings (PyO3 modules). Write them in [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html), including `Parameters`, `Returns`, and `Examples` sections. Keep examples runnable snippets that show real Python usage.
- Rust code must document every public item (`pub` structs, enums, traits, functions, modules). Provide enough context for users to understand invariants and edge cases, and prefer doctests (`/// ```rust
/// let ...
/// ``` `) whenever an example can compile.
- Align the Python and Rust docs: whenever a Rust function backs a Python surface API, ensure the docstring explains the same behavior and, even though cross-linking between Python and Rust docs is usually impossible, call out where the canonical Rust docs live.

## Handy `.justfile` Recipes
Use `just <target>` from the repo root; many tasks pin their working directory for you.

| Target | Purpose |
| --- | --- |
| `develop`, `develop-tests`, `develop-docs`, `develop-all` | Build the local `py-laddu` wheel via `maturin --uv`, optionally adding test/docs extras. |
| `pytest` | Runs Python tests inside `py-laddu` after `develop-tests` (uses `uv run --active pytest`). Prefer this recipe over calling `pytest` directly so the Rust extension is rebuilt first. |
| `test-rs` / `test-rs-docs` | Execute Rust unit/nextest suites and doctests respectively. Combine with `cargo clippy` locally. |
| `test` | Aggregates `pytest`, `test-rs`, and `test-rs-docs` for a full sweep. |
| `builddocs` / `makedocs` | Build the Sphinx HTML docs (dependencies handled via `develop-docs`). |
| `develop-mpi`, `develop-mpi-tests`, `pytest-mpi`, `test-mpi` | Same workflows for the MPI variant under `py-laddu-mpi`. |
| `install-nextest` | Installs `cargo-nextest` (locked) if missing. |
| `clean` | Runs `cargo clean` across the workspace. |

## Working Style Checklist
1. Run the relevant `just develop*` target before touching Python code so the `uv` environment lines up.
2. Keep `ruff` and `cargo clippy` clean alongside the standard Cargo/Python tests (`just test`).
3. When adding APIs, write Rust docs + doctests first, then mirror concise NumPy-style docstrings (with usage examples) on the Python side.
4. Prefer updating `.justfile` recipes rather than ad-hoc commands if you need a repeatable workflow—extend the table above as new tasks appear.

## Git Workflow
- Follow Conventional Commits for every commit message (e.g., `feat: ...`, `fix: ...`).
- Never commit directly to `main`; if an integration branch already exists for the effort, use it.
- When no appropriate branch is available, ask whether to create a new branch or reuse an existing one before proceeding.
