set shell := ["zsh", "-euo", "pipefail", "-c"]

# Show the available development commands.
default:
    @just --list

[private]
require-shell:
    @test -n "${IN_NIX_SHELL:-}" || { echo "enter the project shell with: nix develop" >&2; exit 1; }
    @test -n "${VIRTUAL_ENV:-}" && test -x "$VIRTUAL_ENV/bin/python" || { echo "the uv Python environment is missing; reload the development shell" >&2; exit 1; }

[private]
require-python: require-shell
    @python -c 'import laddu' 2>/dev/null || { echo "Laddu is not installed in the development environment; run: just python-dev" >&2; exit 1; }

# Install/update the main Laddu extension in the uv project environment.
python-dev: require-shell
    maturin develop --manifest-path python/laddu/Cargo.toml --release --generate-stubs

# Install a fast-to-build debug extension in the uv project environment.
python-dev-debug: require-shell
    maturin develop --manifest-path python/laddu/Cargo.toml --generate-stubs

# Install/update the standalone local extension.
python-local: require-shell
    maturin develop --manifest-path python/laddu-local/Cargo.toml --release --generate-stubs

# Install/update the standalone MPI extension.
python-mpi: require-shell
    maturin develop --manifest-path python/laddu-mpi/Cargo.toml --release --generate-stubs

# Install all three Python distributions.
python-all: python-dev python-local python-mpi

# Build release wheels for all three Python distributions.
wheels: require-shell
    uv build --wheel --config-setting maturin.build-args=--generate-stubs python/laddu
    uv build --wheel --config-setting maturin.build-args=--generate-stubs python/laddu-local
    uv build --wheel --config-setting maturin.build-args=--generate-stubs python/laddu-mpi

# Run the practical default closure example (backend: cpu, jit, or gpu).
example backend="cpu" *args: require-python
    python python/examples/closure.py --backend {{backend}} {{args}}

# Run the short closure smoke test.
example-quick backend="cpu" *args: require-python
    python python/examples/closure.py --backend {{backend}} --quick {{args}}

# Run the full Rust-sized closure study.
example-full backend="cpu" *args: require-python
    python python/examples/closure.py --backend {{backend}} --full {{args}}

# Run the practical closure example on the JIT backend.
example-jit *args: require-python
    python python/examples/closure.py --backend jit {{args}}

# Run the practical closure example on the GPU backend.
example-gpu *args: require-python
    python python/examples/closure.py --backend gpu {{args}}

# Print Vulkan information and GPUs visible through Laddu.
gpu-info: require-python
    @vulkaninfo --summary || echo "vulkaninfo could not initialize a device; checking Laddu discovery anyway"
    python -c 'import laddu as ld; devices = ld.gpu.devices(); print("Laddu GPU devices:"); print(*(f"  {device!r}" for device in devices), sep="\n"); raise SystemExit(0 if devices else "no WGPU devices found")'

# Check Rust compilation for every Python distribution.
check-python-rust: require-shell
    cargo check -p laddu-python -p laddu-python-local -p laddu-python-mpi

# Run Rust tests without linking PyO3 extension-module test harnesses.
test-rust: require-shell
    cargo test --workspace --exclude laddu-python --exclude laddu-python-local --exclude laddu-python-mpi
