# Agents Guide

## Tooling Defaults
- Python lives under `py-laddu`/`py-laddu-mpi`; always use `uv` for environment management, installation, and running commands (the `./make.py` entry points wrap `uv pip`, `uv run`, and `maturin` so you rarely need to call them directly).
- Run `ruff` for linting/formatting Python code; prefer `./make.py ruff` (or `uv run ruff ...`) so it shares the same virtual environment that `./make.py develop` maintains.
- For Rust crates, make `cargo clippy --all-targets --all-features` part of your pre-commit check, alongside the existing test targets below.

## Documentation Expectations
- Python docstrings are primarily authored in the Rust bindings (PyO3 modules). Write them in [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html), including `Parameters`, `Returns`, and `Examples` sections. Keep examples runnable snippets that show real Python usage.
- Rust code must document every public item (`pub` structs, enums, traits, functions, modules). Provide enough context for users to understand invariants and edge cases, and prefer doctests (`/// ```rust
/// let ...
/// ``` `) whenever an example can compile.
- Align the Python and Rust docs: whenever a Rust function backs a Python surface API, ensure the docstring explains the same behavior and, even though cross-linking between Python and Rust docs is usually impossible, call out where the canonical Rust docs live.

## Handy `make.py` Commands
Use `./make.py <command>` from the repo root; each command wires up `uv`, Cargo, and other tooling for you.

| Command | Purpose |
| --- | --- |
| `./make.py develop [--tests|--docs|--all-extras] [--mpi] [--python-version X.Y]` | Build the `py-laddu` (or MPI) wheel via `maturin --uv`, optionally adding extras and picking a Python version. |
| `./make.py test [--rust] [--python] [--mpi] [--python-version X.Y]` | Run Rust nextest + doctests and/or Python pytest suites; defaults to both Rust and Python unless flags are provided. |
| `./make.py docs [--no-clean] [--open {python,rust,all}] [--python-version X.Y]` | Build Sphinx docs (and `cargo doc`) with optional cleanup and automatic browser opening. |
| `./make.py develop --mpi ...` / `./make.py test --mpi ...` | Mirror the standard develop/test flows for the MPI frontend. |
| `./make.py ruff [--unsafe-fixes]` / `./make.py ty` | Run linting (`uvx ruff check --fix`) or type checking (`uvx ty check`) inside the managed environment. |
| `./make.py clippy [--mpi]` | Execute `cargo clippy --all-targets`, optionally enabling MPI features. |
| `./make.py clean [--python]` / `./make.py venv [--recreate] [--python-version X.Y]` | Clean the Cargo workspace (and optionally the repo venv) or manage the shared `.venv`. |
| `./make.py docker [--build]` | Build the `laddu:latest` container or drop into a synced shell inside it. |

## Working Style Checklist
1. Run `./make.py develop ...` (add `--tests`/`--docs` as needed) before touching Python code so the managed `uv` environment lines up.
2. Keep `ruff` and `cargo clippy` clean alongside the standard Cargo/Python tests (`./make.py test`). Use `./make.py ruff` and `./make.py ty` to make sure the Python files are clear of type errors and poor syntax choices.
3. When adding APIs, write Rust docs + doctests first, then mirror concise NumPy-style docstrings (with usage examples) on the Python side.
4. Prefer extending `make.py` commands rather than adding ad-hoc scripts if you need a repeatable workflow—keep the table above in sync as new tasks appear.

## Git Workflow
- Follow Conventional Commits for every commit message (e.g., `feat: ...`, `fix: ...`).
- Never commit directly to `main`; if an integration branch already exists for the effort, use it.
- When no appropriate branch is available, ask whether to create a new branch or reuse an existing one before proceeding.
