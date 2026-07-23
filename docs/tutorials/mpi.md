# Distributed analysis with MPI

The MPI distribution exposes the same `import laddu as ld` interface. Under `mpiexec`, every rank builds the same symbolic model while event traversal and reductions are distributed.

```bash
python -m pip install "laddu[mpi]"
mpiexec -n 4 python fit.py
```

## Configure distributed execution

```python
import laddu as ld

execution = ld.Execution(
    "jit",
    precision="f64",
    autodiff="forward",
    mpi=True,
    partitioning="file_groups",
)

data = ld.read_parquet("data/*.parquet", cache="resident")
mc = ld.read_parquet("accepted-mc/*.parquet", cache="resident")
likelihood = ld.Likelihood(
    [ld.NLL(model, data, mc, name="signal")], execution=execution
)
```

Partitioning policies address different storage layouts:

- `contiguous` (also selected by `auto`) assigns contiguous global event ranges.
- `file_groups` assigns source fragments among ranks and is natural for many similarly sized files.
- `rows` distributes strided rows and can balance a small number of uneven files at the cost of less contiguous I/O.

All ranks must enter collective operations in the same order. Keep model construction, likelihood evaluation, fitting, and termination decisions deterministic across ranks.

```python
fit = likelihood.fit(
    ld.ganesh.LBFGSBConfig(history_size=10),
    terminators=[ld.ganesh.MaxSteps(500)],
)

if execution.rank == 0:
    print(fit)
```

## MPI with GPUs

One MPI rank per GPU is a common layout. Select each rank's local device using scheduler-provided locality information or an explicit mapping, then construct `Execution("gpu", device=...)`. Avoid letting every rank choose adapter zero on a multi-GPU node.

Before scaling out, verify a small MPI run against a local `f64` result. Then test rank-count invariance, file coverage, and failure handling. Distributed execution changes summation order, so expect harmless last-bit differences rather than bitwise identity.

