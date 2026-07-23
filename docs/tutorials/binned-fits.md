# Mass-independent binned fits

A mass-independent fit divides an invariant mass $m$ into intervals and fits the angular distribution independently in each interval. No resonance line shape connects neighboring bins. This makes the result a useful intermediate representation for later resonance-model studies.

For angular basis functions $\psi_a(\Omega)$, use

$$
I_b(\Omega)=\left|\psi_0(\Omega)+\sum_{a>0}c_{ab}\psi_a(\Omega)\right|^2,
\qquad c_{ab}=r_{ab}e^{i\phi_{ab}}.
$$

The fixed coefficient of $\psi_0$ removes the unobservable scale and phase convention within each normalized bin.

## Partition data and accepted MC

```python
import laddu as ld

mass = channel.mass("X")
binning = ld.Bin.uniform(30, 1.0, 2.0)
data_bins = data.bin_by(mass, binning, execution=execution)
mc_bins = accepted_mc.bin_by(mass, binning, execution=execution)
```

Both calls return all intervals in order, including empty intervals. Check statistics before constructing likelihood terms.

## Give each bin independent couplings

Assume `s_wave` and `d_wave` are angular expressions without mass-dependent line shapes. Construct one model and one term per populated interval:

```python
terms = []
models = {}

for observed, normalization in zip(data_bins, mc_bins, strict=True):
    if len(observed.dataset) == 0 or len(normalization.dataset) == 0:
        continue

    label = f"bin_{observed.index:02d}"
    radius = ld.parameter(
        f"{label}_d_magnitude", initial=0.2, bounds=(0.0, 5.0), scale=0.5
    )
    phase = ld.parameter(
        f"{label}_d_phase",
        initial=0.0,
        bounds=(-3.141592653589793, 3.141592653589793),
        periodic=True,
    )
    amplitude = s_wave.tagged(f"{label}_s") + ld.polar_complex(radius, phase) * d_wave.tagged(f"{label}_d")
    models[label] = ld.Model(amplitude.norm_sqr())
    terms.append(
        ld.NLL(
            models[label],
            observed.dataset,
            normalization.dataset,
            name=label,
        )
    )

likelihood = ld.Likelihood(terms, execution=execution)
```

Since parameter names are unique by bin, the combined likelihood is a convenient single optimization problem whose gradient spans all intervals. Alternatively fit bins separately to isolate failures and parallelize at the workflow level.

```python
fit = likelihood.fit(
    ld.ganesh.LBFGSBConfig(history_size=10),
    initial=ld.ganesh.VectorInit(likelihood.sample_parameters(seed=9)),
    terminators=[ld.ganesh.MaxSteps(1000)],
)
```

## Report the result

For each interval report its center and width, fitted complex amplitudes or spin-density elements, statistical covariance or bootstrap intervals, fit status, event counts, accepted-MC effective statistics, and the phase convention. Do not draw a smooth resonance curve through bins as though it had been fitted.

```{important}
Low acceptance in one angular region can make an individual mass bin poorly identifiable. Inspect the accepted-MC angular coverage per bin and merge or remove bins using criteria chosen before viewing resonance-like structures.
```

