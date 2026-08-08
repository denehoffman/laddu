---
hide-toc: true
---

# laddu

<div class="hero">
  <img src="_static/logo.svg" alt="The laddu logo">
  <div>
    <p>Amplitude analysis made short and sweet.</p>
  </div>
</div>

laddu brings the full analysis loop into one coherent interface: describe a reaction, compose a differentiable intensity, generate Monte Carlo, fit accepted data, and project the result. Models remain readable Python expressions while evaluation runs on parallel CPU, JIT-compiled CPU, WGPU, or MPI-backed execution.

```python
import laddu as ld

execution = ld.Execution("auto", precision="f64")
likelihood = ld.Likelihood(
    [ld.NLL(model, data=data, accepted_mc=accepted_mc, name="signal")],
    execution=execution,
)
fit = likelihood.fit(
    terminators=[ld.ganesh.MaxSteps(500)],
)
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Learn the workflow
:link: tutorials/index
:link-type: doc
Start with arrays or files, generate samples, build a likelihood, and perform a fit.
:::

:::{grid-item-card} Build physical models
:link: tutorials/polarized-photoproduction
:link-type: doc
Combine topology, helicity amplitudes, line shapes, polarization, and coherent sums.
:::

:::{grid-item-card} Scale an analysis
:link: tutorials/execution
:link-type: doc
Choose precision, automatic differentiation, CPU/JIT/GPU execution, and MPI partitioning.
:::
::::

## Why laddu?

- **One symbolic model.** The same expression drives generation, likelihood evaluation, gradients, and projections.
- **Physics-native building blocks.** Four-vectors, reaction graphs, Wigner functions, relativistic line shapes, and coupled-channel amplitudes are first-class objects.
- **Expression-based architecture.** All mathematical models are built from expressions, so new users can write complex amplitudes in Python and let laddu handle mathematical optimizations, cached evaluations, and batched execution.
- **Reproducible performance choices.** Backends, precision, differentiation strategy, seeds, and partitioning are explicit.
- **Platform-independent execution.** Pipelines are automatically parallel and can be JIT compiled or even run directly on GPUs without writing a single kernel.
- **Distributed programming.** Code written in laddu is fully compatible with the MPI protocol for use on high-performance compute systems.

```{toctree}
:hidden:
:maxdepth: 2

installation
concepts
tutorials/index
reference/api
```
