# Execution backends, memory, and MPI

Every earlier tutorial can use laddu's default execution. Introduce an explicit
{py:class}`laddu.Execution` only when changing performance, precision, memory,
or distribution. The physics model and likelihood API remain unchanged.

## Local backends

```python
automatic = ld.Execution("auto")
cpu = ld.Execution("cpu", threads=8, precision="f64")
jit = ld.Execution("jit", threads=8, precision="f64")
gpu = ld.Execution("gpu", device=0, precision="f32")
```

- `auto` chooses a local strategy and may compile when profitable.
- `cpu` uses the interpreter and avoids compilation startup cost.
- `jit` forces native CPU kernel compilation.
- `gpu` compiles a WGPU compute kernel.

Pass one execution object when constructing the likelihood; the prepared
objective then reuses it throughout optimization and sampling:

```python
likelihood = ld.Likelihood(
    [ld.NLL(model, data=data, accepted_mc=accepted_mc, name="signal")],
    execution=jit,
)
```

Use `f64` as the numerical validation baseline. Before adopting `f32`, compare
objective values, gradients, fitted parameters, and projections on the real
analysis workload.

## Memory planning

Memory budgets accept byte counts, strings, or percentages:

```python
execution = ld.Execution(
    "gpu",
    device=0,
    precision="f32",
    memory=ld.MemoryPlan(
        host="2 GiB",
        device="70% available",
    ),
)
```

Dataset readers and generators also accept memory budgets. Automatic planning
uses smaller chunks when the complete working set does not fit. Inspect
`execution.memory_decisions()` and `execution.memory_report()` after preparing
representative work when diagnosing capacity or chunking.

## Automatic differentiation

```python
forward = ld.Execution("jit", precision="f64", autodiff="forward")
reverse = ld.Execution("gpu", precision="f32", autodiff="reverse")
```

Forward mode carries parameter derivatives with intermediate values; reverse
mode records dependencies from the scalar objective. Parameter count is a
useful first guide, but benchmark complete objective-and-gradient evaluations
after compilation rather than isolated toy expressions.

## Distributed execution with MPI

The MPI distribution exposes the same `import laddu as ld` interface. Every
rank constructs the same model and enters collective operations in the same
order; event traversal and reductions are distributed.

```bash
uv add "laddu[mpi]"
mpiexec -n 4 python analysis.py
```

```python
distributed = ld.Execution(
    "jit",
    precision="f64",
    mpi=True,
    partitioning="file_groups",
)

likelihood = ld.Likelihood(
    [ld.NLL(model, data=data, accepted_mc=accepted_mc, name="signal")],
    execution=distributed,
)

fit = likelihood.fit(
    terminators=[ld.ganesh.MaxSteps(500)],
)
```

Partitioning policies match common storage layouts:

- `contiguous` assigns contiguous global event ranges;
- `file_groups` assigns source fragments and suits many similarly sized files;
- `rows` distributes strided rows when a few files are badly imbalanced.

For GPU clusters, one local MPI rank per GPU is a common layout. Map each rank
to its local device rather than allowing every rank to choose adapter zero.
Validate a small distributed `f64` run against the local result, then check
rank-count invariance and file coverage. Reduction order can change final bits
without changing the statistical result.
