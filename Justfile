set shell := ["zsh", "-euo", "pipefail", "-c"]

project_root := justfile_directory()
python_project := project_root + "/python/laddu"
python_local_project := project_root + "/python/laddu-local"
python_mpi_project := project_root + "/python/laddu-mpi"
python_venv := project_root + "/.venv"
python_free_threaded_venv := project_root + "/.venv-3.14t"

export UV_PROJECT_ENVIRONMENT := python_venv

# Show the available development commands.
default:
    @just --list

[private]
sync-python:
    @command -v uv >/dev/null 2>&1 || { echo "uv is required; install it from https://docs.astral.sh/uv/" >&2; exit 1; }
    uv sync --frozen --inexact --no-install-project --project "{{python_project}}"

[private]
require-python: sync-python
    @uv run --no-sync --project "{{python_project}}" python -c 'import laddu' 2>/dev/null || { echo "laddu is not installed in {{python_venv}}; run: just python-dev" >&2; exit 1; }

[private]
require-docs: require-python
    @uv run --no-sync --project "{{python_project}}" python -c 'import autoapi, myst_parser, sphinx' 2>/dev/null || { echo "the documentation dependencies are missing; run: just docs-install" >&2; exit 1; }

# Install/update the main laddu extension in the uv project environment.
python-dev: sync-python
    cd "{{python_project}}" && VIRTUAL_ENV="{{python_venv}}" UV_PYTHON="{{python_venv}}/bin/python" "{{python_venv}}/bin/maturin" develop --manifest-path Cargo.toml --release --generate-stubs

# Install a fast-to-build debug extension in the uv project environment.
python-dev-debug: sync-python
    cd "{{python_project}}" && VIRTUAL_ENV="{{python_venv}}" UV_PYTHON="{{python_venv}}/bin/python" "{{python_venv}}/bin/maturin" develop --manifest-path Cargo.toml --generate-stubs

# Build the extension and verify its generated type information.
check-python-types: python-dev-debug
    uv run --no-sync --project "{{python_project}}" ty check python docs crates/laddu/examples

# Run the Python binding smoke and concurrency tests.
test-python: python-dev-debug
    uv run --no-sync --project "{{python_project}}" python -m unittest discover -s python/tests -p "test_*.py"

# Build and test under free-threaded CPython 3.14 without re-enabling the GIL.
test-python-free-threaded:
    env -u VIRTUAL_ENV -u PYO3_PYTHON -u PYTHONPATH -u _PYTHON_HOST_PLATFORM -u _PYTHON_SYSCONFIGDATA_NAME UV_PROJECT_ENVIRONMENT="{{python_free_threaded_venv}}" UV_PYTHON=3.14t uv sync --frozen --inexact --no-install-project --project "{{python_project}}"
    cd "{{python_project}}" && env -u PYO3_PYTHON -u PYTHONPATH -u _PYTHON_HOST_PLATFORM -u _PYTHON_SYSCONFIGDATA_NAME VIRTUAL_ENV="{{python_free_threaded_venv}}" UV_PYTHON="{{python_free_threaded_venv}}/bin/python" "{{python_free_threaded_venv}}/bin/maturin" develop --manifest-path Cargo.toml --release --generate-stubs
    env -u VIRTUAL_ENV -u PYO3_PYTHON -u PYTHONPATH -u _PYTHON_HOST_PLATFORM -u _PYTHON_SYSCONFIGDATA_NAME UV_PROJECT_ENVIRONMENT="{{python_free_threaded_venv}}" UV_PYTHON="{{python_free_threaded_venv}}/bin/python" uv run --no-sync --project "{{python_project}}" python -m unittest discover -s python/tests -p "test_*.py"

# Install/update the standalone local extension.
python-local: sync-python
    cd "{{python_local_project}}" && VIRTUAL_ENV="{{python_venv}}" UV_PYTHON="{{python_venv}}/bin/python" "{{python_venv}}/bin/maturin" develop --manifest-path Cargo.toml --release --generate-stubs

# Install/update the standalone MPI extension.
python-mpi: sync-python
    cd "{{python_mpi_project}}" && VIRTUAL_ENV="{{python_venv}}" UV_PYTHON="{{python_venv}}/bin/python" "{{python_venv}}/bin/maturin" develop --manifest-path Cargo.toml --release --generate-stubs

# Install all three Python distributions.
python-all: python-dev python-local python-mpi

# Build release wheels for all three Python distributions.
wheels: sync-python
    uv build --wheel --config-setting maturin.build-args=--generate-stubs python/laddu
    uv build --wheel --config-setting maturin.build-args=--generate-stubs python/laddu-local
    uv build --wheel --config-setting maturin.build-args=--generate-stubs python/laddu-mpi

# Build, install, and verify the primary, local, and MPI distribution artifacts.
test-python-artifacts: wheels
    #!/usr/bin/env zsh
    set -euo pipefail
    contract_venv="$(mktemp -d)"
    trap 'rm -rf "$contract_venv"' EXIT
    uv venv "$contract_venv"
    uv pip install --python "$contract_venv/bin/python" python/laddu/dist/*.whl python/laddu-local/dist/*.whl
    "$contract_venv/bin/python" python/tests/installed_artifact_contract.py --module laddu --backend local --distribution laddu
    "$contract_venv/bin/python" python/tests/installed_artifact_contract.py --module _laddu_local --backend local --distribution laddu-local
    uv pip install --python "$contract_venv/bin/python" python/laddu-mpi/dist/*.whl
    "$contract_venv/bin/python" python/tests/installed_artifact_contract.py --module _laddu_mpi --backend mpi --distribution laddu-mpi
    "$contract_venv/bin/python" python/tests/installed_artifact_contract.py --module laddu --backend mpi --distribution laddu

# Install the Sphinx documentation dependencies in the development environment.
docs-install: sync-python
    uv pip install --python "{{python_venv}}/bin/python" -r docs/requirements.txt

# Build the Markdown/Sphinx documentation with warnings treated as errors.
docs-build: require-docs
    uv run --no-sync --project "{{python_project}}" sphinx-build -E -W --keep-going -b html docs docs/_build/html

# Build and serve the Markdown/Sphinx documentation locally.
docs-serve port="8000": docs-build
    @echo "serving laddu documentation at http://127.0.0.1:{{port}}"
    uv run --no-sync --project "{{python_project}}" python -m http.server {{port}} --bind 127.0.0.1 --directory docs/_build/html

# Build API documentation for every Rust crate without dependency documentation.
rust-docs-build:
    cargo doc --workspace --no-deps --exclude laddu-python --exclude laddu-python-local --exclude laddu-python-mpi

# Build the Rust API documentation and open it in the default browser.
rust-docs-open:
    cargo doc --workspace --no-deps --open --exclude laddu-python --exclude laddu-python-local --exclude laddu-python-mpi

# Run the practical default closure example (backend: cpu, jit, or gpu).
example backend="cpu" *args: require-python
    uv run --no-sync --project "{{python_project}}" python python/examples/closure.py --backend {{backend}} {{args}}

# Run the short closure smoke test.
example-quick backend="cpu" *args: require-python
    uv run --no-sync --project "{{python_project}}" python python/examples/closure.py --backend {{backend}} --quick {{args}}

# Run the full Rust-sized closure study.
example-full backend="cpu" *args: require-python
    uv run --no-sync --project "{{python_project}}" python python/examples/closure.py --backend {{backend}} --full {{args}}

# Run the practical closure example on the JIT backend.
example-jit *args: require-python
    uv run --no-sync --project "{{python_project}}" python python/examples/closure.py --backend jit {{args}}

# Run the practical closure example on the GPU backend.
example-gpu *args: require-python
    uv run --no-sync --project "{{python_project}}" python python/examples/closure.py --backend gpu {{args}}

# Run the four-period acceptance and differential-cross-section example.
cross-section-example backend="cpu" *args: require-python
    uv run --no-sync --project "{{python_project}}" python python/examples/cross_section.py --backend {{backend}} {{args}}

# Run the short cross-section smoke test.
cross-section-example-quick backend="cpu" *args: require-python
    uv run --no-sync --project "{{python_project}}" python python/examples/cross_section.py --backend {{backend}} --quick {{args}}

# Run the higher-statistics cross-section study.
cross-section-example-full backend="cpu" *args: require-python
    uv run --no-sync --project "{{python_project}}" python python/examples/cross_section.py --backend {{backend}} --full {{args}}

# Print Vulkan information and GPUs visible through laddu.
gpu-info: require-python
    @vulkaninfo --summary || echo "vulkaninfo could not initialize a device; checking laddu discovery anyway"
    uv run --no-sync --project "{{python_project}}" python -c 'import laddu as ld; devices = ld.gpu.devices(); print("laddu GPU devices:"); print(*(f"  {device!r}" for device in devices), sep="\n"); raise SystemExit(0 if devices else "no WGPU devices found")'

# Check Rust compilation for every Python distribution.
check-python-rust:
    cargo check -p laddu-python -p laddu-python-local -p laddu-python-mpi

# Run Rust tests without linking PyO3 extension-module test harnesses.
test-rust:
    cargo test --workspace --exclude laddu-python --exclude laddu-python-local --exclude laddu-python-mpi
