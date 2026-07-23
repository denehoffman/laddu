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
    [ld.NLL(model, data, accepted_mc, name="signal")],
    execution=execution,
)
fit = likelihood.fit(
    ld.ganesh.LBFGSBConfig(history_size=10),
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
- **Acceptance-aware inference.** Normalization is evaluated over accepted Monte Carlo, so detector effects enter directly.
- **Physics-native building blocks.** Four-vectors, reaction graphs, Wigner functions, relativistic line shapes, and coupled-channel amplitudes are first-class objects.
- **Reproducible performance choices.** Backends, precision, differentiation strategy, seeds, and partitioning are explicit.

```{toctree}
:hidden:
:maxdepth: 2

installation
concepts
tutorials/index
reference/api
```
