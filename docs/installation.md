# Installation

laddu supports Python 3.11 and newer.

## Standard installation

```bash
python -m pip install laddu
```

The standard distribution includes local CPU, JIT, and WGPU support. Inspect what is available on the current machine rather than assuming a device exists:

```python
import laddu as ld

print(ld.backend())
print(ld.capabilities())
print(ld.gpu.devices())
```

## MPI installation

Install an MPI implementation and its development headers, then install the MPI extra:

```bash
python -m pip install "laddu[mpi]"
mpiexec -n 4 python analysis.py
```

All ranks execute the same script. Dataset traversal, reductions, likelihood values, and gradients are distributed; guard rank-zero-only output with `execution.rank == 0`.

## Development checkout

Install Rust, `uv`, and `just`, then use the repository recipes directly:

```bash
git clone https://github.com/denehoffman/laddu.git
cd laddu
just python-dev
just example-quick cpu
```

The recipes create and use a project-local `.venv`. The optional `nix develop`
shell provides the toolchain and native MPI, Vulkan, and GPU dependencies, but
the recipes do not depend on Nix.

`just python-dev-debug` shortens edit/build cycles. Use a release build before benchmarking.

```{tip}
Record `laddu.backend()`, `laddu.capabilities()`, `repr(execution)`, package versions, and random seeds with every analysis result.
```
