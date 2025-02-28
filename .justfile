default:
  just --list

[working-directory: 'py-laddu/laddu-std']
develop:
  CARGO_INCREMENTAL=true maturin develop --uv

[working-directory: 'py-laddu/laddu-mpi']
develop-mpi:
  CARGO_INCREMENTAL=true maturin develop --uv

builddocs:
  CARGO_INCREMENTAL=true maturin build -m py-laddu/Cargo.toml
  uv pip install ./target/wheels/*
  make -C py-laddu/docs clean
  make -C py-laddu/docs html

makedocs:
  make -C py-laddu/docs clean
  make -C py-laddu/docs html

odoc:
  firefox ./py-laddu/docs/build/html/index.html

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
