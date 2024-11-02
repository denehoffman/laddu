default:
  just --list

develop:
  CARGO_INCREMENTAL=true maturin develop -r --uv --strip

pydoc:
  make -C docs clean
  make -C docs html

odoc:
  firefox ./docs/build/html/index.html
