default:
  just --list

develop:
  CARGO_INCREMENTAL=true maturin develop -r --uv --strip

builddocs:
  CARGO_INCREMENTAL=true maturin build -r --strip
  uv pip install ./target/wheels/*
  make -C docs clean
  make -C docs html

makedocs:
  make -C docs clean
  make -C docs html

odoc:
  firefox ./docs/build/html/index.html

clean:
  cargo clean

profile:
  RUSTFLAGS='-C force-frame-pointers=y' cargo build --profile perf
  perf record -g target/perf/laddu
  perf annotate -v --asm-raw --stdio
  perf report -g graph,0.5,caller

popen:
  mv firefox.perf.data firefox.perf.data.old
  perf script --input=perf.data -F +pid > firefox.perf.data
  firefox https://profiler.firefox.com
