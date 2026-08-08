# Fitting models across bins of event data

Here the data and accepted MC are partitioned by an observable, and an
independent angular model is fitted in each interval. The model itself is not a
"binned model": the likelihood still evaluates individual events within every
bin.

A mass-independent analysis is a common example. With angular basis functions
$\psi_a(\Omega)$,

$$
I_b(\Omega)=\left|\psi_0(\Omega)
+\sum_{a>0}c_{ab}\psi_a(\Omega)\right|^2,
\qquad c_{ab}=r_{ab}e^{i\phi_{ab}},
$$

and each mass interval $b$ has its own complex coefficients.

## Partition observed and normalization data

```python
mass = generation_channel.mass("X")
mass_edges = np.linspace(1.0, 2.0, 31, dtype=np.float32)
binning = ld.Bin(mass_edges)

data_bins = data.bin_by(mass, bins=binning)
accepted_bins = accepted_mc.bin_by(mass, bins=binning)
```

Both results contain every interval in order. Decide how to handle empty or
low-statistics bins before looking for structures.

## Construct one event likelihood per interval

Assume `s_wave` and `d_wave` are angular expressions without a mass-dependent
line shape:

```python
terms = []

for observed, normalization in zip(data_bins, accepted_bins, strict=True):
    if len(observed.dataset) == 0 or len(normalization.dataset) == 0:
        continue

    prefix = f"bin_{observed.index:02d}"
    magnitude = ld.parameter(
        f"{prefix}_d_magnitude",
        initial=0.2,
        bounds=(0.0, 5.0),
    )
    phase = ld.parameter(
        f"{prefix}_d_phase",
        initial=0.0,
        bounds=(-3.141592653589793, 3.141592653589793),
        periodic=True,
    )

    amplitude = s_wave + ld.polar_complex(magnitude, phase) * d_wave
    bin_model = ld.Model(amplitude.norm_sqr())
    terms.append(
        ld.NLL(
            bin_model,
            data=observed.dataset,
            accepted_mc=normalization.dataset,
            name=prefix,
        )
    )

likelihood = ld.Likelihood(terms)
fit = likelihood.fit(
    initial=likelihood.sample_parameters(seed=9),
    terminators=[ld.ganesh.MaxSteps(1000)],
)
```

Unique parameter names make this one block-diagonal optimization problem.
Fitting bins separately is also reasonable when failure isolation or workflow
parallelism matters.

Report each interval's edges, event and effective accepted-MC statistics,
fitted complex coefficients, uncertainty, fit status, and phase convention.
Do not draw a smooth resonance curve through the points as though a resonance
line shape had been fitted.
