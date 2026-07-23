# From fitted intensities to cross sections

An amplitude fit normally determines shapes and relative complex couplings. An absolute cross section additionally requires integrated luminosity, efficiency, branching fractions, bin widths, and a yield convention.

For bin $k$,

$$
\left.\frac{d\sigma}{dx}\right|_k
=\frac{N_k^\mathrm{signal}}
{\mathcal L_\mathrm{int}\,\epsilon_k\,\mathcal B\,\Delta x_k}.
$$

Here $N_k^\mathrm{signal}$ may come from an extended likelihood or from acceptance-corrected fitted projection weights. Keep the convention explicit: coherent totals include interference and generally do not equal the sum of single-component yields.

## Efficiency from matching MC samples

If generated and accepted samples represent the same thrown distribution,

$$\epsilon_k=\frac{\sum_{j\in\mathrm{accepted},k}w_j}{\sum_{j\in\mathrm{generated},k}w_j}.$$

```python
import numpy as np

generated_bins = generated_mc.bin_by(x, binning, execution=execution)
accepted_bins = accepted_mc.bin_by(x, binning, execution=execution)
efficiency = np.array([
    a.dataset.sum_weights() / g.dataset.sum_weights()
    for g, a in zip(generated_bins, accepted_bins, strict=True)
])
```

If generation was importance sampled, use the proper generator weights in both numerator and denominator. Evaluate uncertainty from finite MC, luminosity, background subtraction, branching fractions, model dependence, and fit statistics.

## Projection-based yields

```python
projection = likelihood.projection("signal", generated_mc, ["reference", "second"])
projection_weights = np.asarray(
    projection.weights(fitted, acceptance_corrected=True), dtype=float
)
```

Histogram these weights in the reporting variable and establish their normalization against a known generated yield or an extended-likelihood yield. A normalized `NLL` alone cannot determine an absolute scale because $I\mapsto cI$ cancels from its objective.

For publication, archive the luminosity inputs, efficiency maps, response variations, selection definition, fit configuration, and code version alongside the numerical cross-section table.

