default:
  just --list

develop:
  maturin develop -r --uv --strip
