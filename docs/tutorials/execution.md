# Execution and compilation settings

An `Execution` object makes performance and numerical choices explicit. Pass the same object to generation, dataset evaluation, binning, and likelihood preparation.

## Backend selection

```python
import laddu as ld

automatic = ld.Execution("auto")
cpu = ld.Execution("cpu", threads=8, precision="f64", memory="60% available")
jit = ld.Execution("jit", threads=8, precision="f64")
gpu = ld.Execution(
    "gpu",
    device=0,
    precision="f32",
    memory=ld.MemoryPlan(host="2 GiB", device="70% available"),
)
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

GPU selection also accepts a name or PCI bus ID.

## Memory planning

Memory budgets are the primary control for data loading, compiled caches,
generation, and CPU/GPU evaluation. They accept byte counts, strings such as
`"8 GiB"`, or portable percentages such as `"70% total"` and
`"60% available"`. Automatic planning uses less memory when the complete
working set is smaller than the request.

```python
state = ld.MemoryState.current()
state.refresh()
print(state.host.total_bytes, state.host.available_bytes)

execution = ld.Execution("jit", memory="50% available")
# Prepare/evaluate work, then inspect the resolved strategy and chunk size.
print(execution.memory_decisions())
print(execution.memory_report())
```

The report includes current process RSS and virtual memory plus a sampled RSS
high-water mark. These operating-system counters cover the complete process;
the per-resource `reserved_bytes` and `high_water_bytes` fields separately
show allocations tracked by laddu's budget pools.

For MPI jobs, laddu shares a node budget among local ranks using launcher
metadata. If that metadata is unavailable it conservatively shares across the
whole communicator. Reports include the resolved MPI sharing policy.

GPU discovery prefers NVIDIA NVML when the runtime library is installed. It
otherwise uses DXGI's per-process video-memory budget on Windows, Metal's
recommended working-set size on macOS, or DRM/sysfs on Linux. When none of
those providers exposes capacity, laddu uses a clearly labeled adaptive
buffer-limit estimate. Telemetry failures are non-fatal, and shared systems can
supply authoritative capacity before constructing an execution:

```python
state.set_device_capacity(
    "pci:0000:65:00.0", 24 * 1024**3, available_bytes=18 * 1024**3
)
```

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
