venv := ".venv"
bin := ".venv/bin"
python := ".venv/bin/python"
export CARGO_INCREMENTAL := "1"
export UV_PYTHON := ".venv/bin/python"
set quiet

# Choose the command to run
default:
    just --choose

# Create a venv if it doesn't exist
create-venv:
    if [ ! -d "{{venv}}" ]; then uv venv {{venv}} --python=3.14; fi

# Build the laddu-cpu wheel
build-python-cpu: create-venv
    uvx --with "maturin[patchelf]>=1.7,<2" maturin build --manifest-path py-laddu-cpu/Cargo.toml --release -o py-laddu-cpu/dist

# Build the laddu-mpi wheel
build-python-mpi: create-venv
    uvx --with "maturin[patchelf]>=1.7,<2" maturin build --manifest-path py-laddu-mpi/Cargo.toml --release -o py-laddu-mpi/dist

# Build both laddu-cpu and laddu-mpi wheels
build-python: build-python-cpu build-python-mpi

# Install laddu (Python, CPU)
develop: build-python-cpu
    uv pip install --find-links py-laddu-cpu/dist -e py-laddu

# Install laddu (Python, CPU, tests)
develop-tests: build-python-cpu
    uv pip install --find-links py-laddu-cpu/dist -e "py-laddu[tests]"

# Install laddu (Python, CPU, MPI)
develop-mpi: build-python
    uv pip install --find-links py-laddu-cpu/dist --find-links py-laddu-mpi/dist -e "py-laddu[mpi]"

# Install laddu (Python, CPU, MPI, tests)
develop-tests-mpi: build-python
    uv pip install --find-links py-laddu-cpu/dist --find-links py-laddu-mpi/dist -e "py-laddu[tests,mpi]"

# Test Python library (CPU)
test-python: develop-tests
    {{bin}}/pytest

# Test Python library (MPI)
test-python-mpi: develop-tests-mpi
    LADDU_BACKEND="MPI" {{bin}}/pytest

# Test Rust crate (CPU)
test-rust:
    cargo nextest run
    cargo test --doc

# Test Rust crate (MPI)
test-rust-mpi:
    cargo nextest run --features mpi
    cargo test --doc --features mpi

# Test Rust and Python (CPU)
test: test-rust test-python

# Test Rust and Python (MPI)
test-mpi: test-rust-mpi test-python-mpi

# Run Python lints and type checking
lint-python:
    ruff check py-laddu
    ruff format --check py-laddu
    ty check

# Run Rust lints and checks (CPU)
lint-rust:
    cargo clippy --all-targets

# Run Rust lints and checks (MPI)
lint-rust-mpi:
    cargo clippy --all-targets --features mpi

# Run Rust (CPU) and Python lints and checks
lint: lint-rust lint-python

# Run Rust (MPI) and Python lints and checks
lint-mpi: lint-rust-mpi lint-python

# Build Rust crates (CPU)
build-rust:
    cargo build --all-targets

# Build Rust crates (MPI)
build-rust-mpi:
    cargo build --all-targets --features mpi

# Build Rust crates and Python wheels (CPU)
build: build-rust build-python-cpu

# Build Rust crates and Python wheels (MPI)
build-mpi: build-rust-mpi build-python-mpi

# Clean Rust targets
clean-rust:
    cargo clean

# Clean Python targets
clean-python:
    rm -rf .venv .uv-cache dist py-laddu/build py-laddu-cpu/dist py-laddu-mpi/dist

# Clean Rust and Python targets
clean: clean-rust clean-python

# Build Python documentation
docs-python: build-python-cpu
    uv pip install --find-links py-laddu-cpu/dist -e "py-laddu[docs]"
    make -C py-laddu/docs html

# Build Rust documentation
docs-rust:
    cargo doc --all-features

# Build Rust and Python documentation
docs: docs-rust docs-python

# Run cargo-hack check over each supported Rust version and feature combination
hack-check:
    cargo hack check --rust-version --each-feature --no-dev-deps

# Run cargo-hack test over each feature combination
hack-test:
    cargo hack test --each-feature

# Run cargo-hack check and test
hack: hack-check hack-test

# Build the docker development image
docker-build:
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT_DIR=$(pwd)
    IMAGE=${LADDU_DOCKER_IMAGE:-laddu:latest}
    docker build -t "$IMAGE" "$ROOT_DIR"

# Enter the development image shell
docker-shell:
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT_DIR=$(pwd)
    IMAGE=${LADDU_DOCKER_IMAGE:-laddu:latest}
    SRC=/src
    WORKDIR=/work
    FLAG=LADDU_INSIDE_DOCKER
    cmd=(
      docker run --rm
      -v "$ROOT_DIR:$SRC:ro"
      --tmpfs "$WORKDIR:rw,exec,nosuid,size=8g"
      --tmpfs "$WORKDIR/.venv:rw,exec,nosuid,size=2g"
      -w "$WORKDIR"
      -e "$FLAG=1"
      -e "LD_LIBRARY_PATH=$WORKDIR/.venv/lib:$LD_LIBRARY_PATH"
    )
    if [[ -t 0 ]]; then cmd+=(-i); fi
    if [[ -t 1 ]]; then cmd+=(-t); fi

    INNER=$(cat <<'SH'
    set -euo pipefail
    mkdir -p /work
    rsync -a --delete --exclude 'target/' --exclude '.venv/' --exclude '__pycache__/' /src/ /work/
    SYSTEM_CONFIGURATION_DISABLE=1 UV_CACHE_DIR=/work/.uv-cache uv venv /work/.venv
    PYTHON_LIB=$(find /root/.local/share/uv/python -maxdepth 2 -type d -name "cpython-*-linux-*-gnu" | head -n 1)/lib || true
    export LD_LIBRARY_PATH=$PYTHON_LIB:$LD_LIBRARY_PATH
    cd /work
    exec bash
    SH
    )
    cmd+=("$IMAGE" "bash" "-lc" "$INNER")
    exec "${cmd[@]}"

# Sync laddu's pyproject.toml's version with workspace since it isn't a Rust crate
sync-versions: create-venv
    #!/usr/bin/env -S uv run --script
    # /// script
    # requires-python = ">=3.11"
    # ///
    from __future__ import annotations

    import re
    from pathlib import Path
    import tomllib

    root = Path.cwd()
    workspace = tomllib.loads((root / 'Cargo.toml').read_text())
    version = workspace['workspace']['dependencies']['laddu']['version']

    wrapper_path = root / 'py-laddu' / 'pyproject.toml'
    text = wrapper_path.read_text()

    def replace_once(source: str, pattern: str, replacement: str) -> str:
        compiled = re.compile(pattern, re.MULTILINE)
        new_text, count = compiled.subn(replacement, source, count=1)
        if count == 0:
            msg = f'Pattern {pattern!r} not found in {wrapper_path}'
            raise RuntimeError(msg)
        return new_text

    text = replace_once(text, r'^version\s*=\s*".*"$', f'version = "{version}"')
    text = replace_once(text, r'^\s*"laddu-cpu ==[^\"]+",$', f'  "laddu-cpu == {version}",')
    text = replace_once(text, r'^mpi = \["laddu-mpi ==[^\"]+"\]$', f'mpi = ["laddu-mpi == {version}"]')
    wrapper_path.write_text(text)
    print(f"Successfully synced versions to {version}")

# Run Rust coverage analysis
coverage-rust:
    cargo llvm-cov --workspace --lcov --output-path coverage-rust.lcov --summary-only --exclude-from-report py-laddu

# Run Python coverage analysis
coverage-python: develop-tests
    {{bin}}/pytest --cov --cov-report xml:coverage-python.xml

# Run Rust and Python coverage analysis
coverage: coverage-rust coverage-python
