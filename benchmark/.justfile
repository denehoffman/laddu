bench-build:
  docker build -t benchmark .

bench-shell:
  docker run -it -v "$(pwd)/..:/work" benchmark
