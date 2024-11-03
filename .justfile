default:
  just --list

develop:
  CARGO_INCREMENTAL=true maturin develop -r --uv --strip

makedocs:
  CARGO_INCREMENTAL=true maturin build -r --strip
  uv pip install ./target/wheels/*
  make -C docs clean
  make -C docs html

odoc:
  firefox ./docs/build/html/index.html
