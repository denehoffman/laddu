build:
  cargo build --release

mprofile: build
  valgrind --tool=massif --time-unit=ms --stacks=yes --trace-children=yes --massif-out-file=massif.out cargo run --release --quiet

mplot: mprofile
  ./scripts/plot_massif.py massif.out memory_profile.png
