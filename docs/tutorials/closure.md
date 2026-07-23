# Capstone: a complete closure analysis

The repository includes a full $\gamma p\to K_S^0K_S^0p$ closure study. It builds a sequential-helicity $f_0(1500)+f_2(1270)$ model, generates pseudo-data and normalization MC, fits two free parameters, performs Poisson-bootstrap refits, plots tagged projections, and writes Parquet plus JSON outputs.

The fitted intensity is schematically

$$
I(\Omega)=\frac14\sum_{\lambda_\gamma,\lambda_t,\lambda_r}
\left|\mathcal A_{0}(\Omega)
+r_2e^{i\phi_2}\mathcal A_{2}(\Omega)\right|^2.
$$

From the development shell:

```bash
just python-dev
just example-quick cpu
just example gpu
just example-full jit
```

The quick mode is an API smoke test, the default is practical for iteration, and full mode increases event and bootstrap counts. Outputs are written below `target/python-closure` by default.

## Read the example in stages

1. `build_channel` defines the reaction graph and proposal distributions.
2. `sequential_wave` constructs production and decay rotations with explicit spin coupling.
3. `build_model` forms the coherent $S$/$D$ intensity and tags both contributions.
4. `main` generates model pseudo-data and phase-space normalization MC.
5. `fit_likelihood` prepares bounded L-BFGS-B optimization.
6. `plot_closure` refits bootstrap replicas and produces a projection band.

```{literalinclude} ../../python/examples/closure.py
:language: python
:caption: python/examples/closure.py
:linenos:
```

## Closure criteria

Do not judge closure from a single best-fit point. Across independent pseudo-experiments, check bias, pull mean and width, confidence-interval coverage, fit failure rate, boundary frequency, and projection residuals. Repeat with accepted MC processed through the intended detector and selection chain.

The example is also a useful backend regression: compare CPU, JIT, and GPU objective values and fitted parameters with tolerances appropriate to their precision.

