default:
  just --list

develop:
  CARGO_INCREMENTAL=true maturin develop -r --uv --strip
