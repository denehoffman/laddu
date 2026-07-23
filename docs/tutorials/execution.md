# Execution and compilation settings

An `Execution` object makes performance and numerical choices explicit. Pass the same object to generation, dataset evaluation, binning, and likelihood preparation.

## Backend selection

```python
import laddu as ld

automatic = ld.Execution("auto")
cpu = ld.Execution("cpu", threads=8, precision="f64")
jit = ld.Execution("jit", threads=8, precision="f64")
gpu = ld.Execution("gpu", device=0, precision="f32")
```

- `auto` uses the local CPU and may JIT compile when profitable.
- `cpu` retains the interpreter and avoids JIT startup cost.
- `jit` forces native CPU kernel compilation; it is often useful when the prepared model is evaluated many times.
- `gpu` compiles a WGPU compute kernel. It is most attractive for large batches and arithmetic-heavy models.

Inspect adapters before choosing by index or name:

```python
for device in ld.gpu.devices():
    print(device.index, device.name, device.supports_f64, device.max_buffer_size)
```

GPU selection also accepts a name or PCI bus ID. `memory_budget` limits allocations for shared systems.

## Precision and differentiation

```python
forward64 = ld.Execution("jit", precision="f64", autodiff="forward")
reverse32 = ld.Execution("gpu", precision="f32", autodiff="reverse")
```

Use `f64` as the validation baseline. Compare objective values, gradients, fitted parameters, and projections before adopting `f32`. Not every GPU supports double-precision shaders.

Forward mode propagates a derivative vector with intermediate values; reverse mode records and traverses dependencies from the scalar objective. Parameter count is a useful guide, not a substitute for measurement.

## A reproducible benchmark

Warm up compilation separately, then time repeated evaluations of the same prepared likelihood:

```python
import time
import numpy as np

likelihood = ld.Likelihood([ld.NLL(model, data, mc)], execution=execution)
x = likelihood.default_parameters
likelihood.value_and_gradient(x)  # preparation/warm-up

started = time.perf_counter()
for _ in range(20):
    likelihood.value_and_gradient(x)
elapsed = time.perf_counter() - started
print(f"{elapsed / 20:.4f} s/evaluation")
```

Benchmark the complete workload with realistic event counts and model complexity. Small examples are commonly dominated by compilation, transfer, or scheduling overhead.

```{note}
Models are compiled when a model-backed likelihood is prepared. Construct the likelihood once, then reuse it throughout minimization and sampling.
```

