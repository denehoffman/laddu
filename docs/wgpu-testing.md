# WGPU test policy

The `laddu-wgpu` tests are split by dependency:

- `--lib` runs the hardware-independent unit modules in
  `src/scalar/tests.rs` and generated-shader validation in
  `src/scalar/shader_tests.rs`; these tests do not open an adapter.
- `--test hardware -- --ignored` is an adapter/device integration suite. It
  is opt-in and must run only on a runner with a known WGPU-compatible
  adapter. A missing adapter is an infrastructure issue, not a product-test
  failure.

Ordinary CI must not make hardware availability a prerequisite for merging.
When a hardware runner is available, run the ignored suite explicitly:

```text
cargo test -p laddu-wgpu --test hardware -- --ignored
```

The existing ignored runtime reduction test follows the same policy. Keep
shader-generation and memory-geometry characterization tests in the normal
suite; reserve ignored tests for actual device creation, queue submission, or
GPU result readback.
