# List all available recipes
default:
  just --list

# Develop local version of laddu in py-laddu using maturin and uv
[working-directory: 'py-laddu']
develop:
  uv pip uninstall laddu
  CARGO_INCREMENTAL=true maturin develop --uv

# Develop laddu with tests dependencies in py-laddu
[working-directory: 'py-laddu']
develop-tests:
  uv pip uninstall laddu
  CARGO_INCREMENTAL=true maturin develop --uv --extras=tests

# Develop laddu with documentation dependencies in py-laddu
[working-directory: 'py-laddu']
develop-docs:
  uv pip uninstall laddu
  CARGO_INCREMENTAL=true maturin develop --uv --extras=docs

# Develop laddu with both test and doc dependencies in py-laddu
[working-directory: 'py-laddu']
develop-all:
  uv pip uninstall laddu
  CARGO_INCREMENTAL=true maturin develop --uv --extras=tests,docs

# Run Python tests after building with test extras
[working-directory: 'py-laddu']
pytest: develop-tests
  uv run --active pytest

# Run Rust tests using cargo-nextest
test-rs:
  cargo nextest run

# Run Rust documentation tests
test-rs-docs:
  cargo test --doc

# Run all tests: Python, Rust, and Rust doc tests
test: develop-tests test-rs test-rs-docs pytest
  @true

# Build HTML documentation using Sphinx
[working-directory: 'py-laddu']
builddocs: develop-docs
  make -C docs clean
  make -C docs html

# Alias for building HTML documentation
[working-directory: 'py-laddu']
makedocs: develop-docs
  make -C docs clean
  make -C docs html

# Develop local version of laddu-mpi in py-laddu-mpi
[working-directory: 'py-laddu-mpi']
develop-mpi:
  uv pip uninstall laddu
  CARGO_INCREMENTAL=true maturin develop --uv

# Develop laddu-mpi with tests dependencies
[working-directory: 'py-laddu-mpi']
develop-mpi-tests:
  uv pip uninstall laddu
  CARGO_INCREMENTAL=true maturin develop --uv --extras=tests

# Run Python tests for MPI variant
[working-directory: 'py-laddu-mpi']
pytest-mpi: develop-mpi-tests
  uv run --active pytest

# Run all MPI-related tests
[working-directory: 'py-laddu-mpi']
test-mpi: pytest-mpi
  @true

# Install cargo-nextest if not already installed
install-nextest:
  cargo install cargo-nextest --locked

# Clean all cargo build artifacts
clean:
  cargo clean
