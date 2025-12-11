venv := ".venv"
bin := ".venv/bin"
python := ".venv/bin/python"
export CARGO_INCREMENTAL := "1"
export UV_PYTHON := ".venv/bin/python"
set quiet

default:
    just --choose

create-venv:
    if [ ! -d "{{venv}}" ]; then uv venv {{venv}}; fi

build-cpu: create-venv
    uvx --with "maturin[patchelf]>=1.7,<2" maturin build --manifest-path py-laddu-cpu/Cargo.toml --release -o py-laddu-cpu/dist

build-mpi: create-venv
    uvx --with "maturin[patchelf]>=1.7,<2" maturin build --manifest-path py-laddu-mpi/Cargo.toml --release -o py-laddu-mpi/dist

build-python: build-cpu build-mpi

develop: build-cpu
    uv pip install --find-links py-laddu-cpu/dist -e py-laddu

develop-tests: build-cpu
    uv pip install --find-links py-laddu-cpu/dist -e "py-laddu[tests]"

develop-mpi: build-python
    uv pip install --find-links py-laddu-cpu/dist --find-links py-laddu-mpi/dist -e "py-laddu[mpi]"

develop-tests-mpi: build-python
    uv pip install --find-links py-laddu-cpu/dist --find-links py-laddu-mpi/dist -e "py-laddu[tests,mpi]"


test-python: develop-tests
    {{bin}}/pytest

test-python-mpi: develop-tests-mpi
    LADDU_BACKEND="MPI" {{bin}}/pytest

test-rust:
    cargo nextest run
    cargo test --doc

test-rust-mpi:
    cargo nextest run --features mpi
    cargo test --doc --features mpi

test-all: test-rust test-python

test-all-mpi: test-rust-mpi test-python-mpi

lint-python:
    uv tool add ruff
    uv tool add ty
    {{bin}}/ruff check py-laddu
    {{bin}}/ruff format --check py-laddu
    {{bin}}/ty check

lint-rust:
    cargo clippy --all-targets

lint-rust-mpi:
    cargo clippy --all-targets --features mpi

lint-all: lint-rust lint-python

lint-all-mpi: lint-rust-mpi lint-python

build-rust:
    cargo build --all-targets

build-rust-mpi:
    cargo build --all-targets --features mpi

build-all: build-rust build-cpu

build-all-mpi: build-rust-mpi build-mpi

clean-rust:
    cargo clean

clean-python:
    rm -rf .venv .uv-cache dist py-laddu/build py-laddu-cpu/dist py-laddu-mpi/dist

clean-all: clean-rust clean-python

docs-python: build-cpu
    uv pip install --find-links py-laddu-cpu/dist -e "py-laddu[docs]"
    make -C py-laddu/docs html

docs-rust:
    cargo doc --all-features

docs-all: docs-rust docs-python

hack-check:
    cargo hack check --rust-version --each-feature --no-dev-deps

hack-test:
    cargo hack test --each-feature

hack: hack-check hack-test

docker-build:
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT_DIR=$(pwd)
    IMAGE=${LADDU_DOCKER_IMAGE:-laddu:latest}
    docker build -t "$IMAGE" "$ROOT_DIR"

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

sync-versions:
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

coverage-rust:
    cargo llvm-cov --workspace --lcov --output-path coverage-rust.lcov --summary-only --exclude-from-report py-laddu

coverage-python: develop-tests
    {{bin}}/pytest --cov --cov-report xml:coverage-python.xml

coverage-all: coverage-rust coverage-python
