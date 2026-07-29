# Generating modeled Monte Carlo

The same generator can evaluate a model on each proposal. This produces either weighted model MC or unit-weight pseudo-data.

## Define a small intensity

```python
import laddu as ld

s = channel.s("X")
m_k = channel.particle("ks1").mass
bw = ld.relativistic_breit_wigner(s, 1.50, 0.11, m_k, m_k, l=0)
magnitude = ld.parameter("magnitude", initial=0.5, bounds=(0.0, 2.0))
phase = ld.parameter("phase", initial=0.0, bounds=(-3.14159, 3.14159), periodic=True)
amplitude = bw + ld.polar_complex(magnitude, phase)
model = ld.Model(amplitude.norm_sqr())
truth = {"magnitude": 0.7, "phase": 0.9}
```

Weighted generation is efficient when downstream code accepts weights:

```python
weighted, report = generator.weighted(
    100_000,
    model=model,
    parameters=truth,
    execution=execution,
    seed=20,
)
```

## Accept–reject unweighting

If $q(\Omega)$ is the proposal and $w(\Omega)$ the target-to-proposal weight, accept with

$$P_\mathrm{accept}(\Omega)=\frac{w(\Omega)}{W},\qquad W\geq\sup_\Omega w(\Omega).$$

```python
pseudo_data, report = generator.unweighted(
    10_000,
    model,
    parameters=truth,
    execution=execution,
    seed=21,
    pilot_proposals=50_000,
    safety_factor=2.0,
    grow_envelope=True,
    max_proposals=50_000_000,
)
print(f"acceptance: {report.acceptance_rate:.2%}")
```

A pilot estimate is convenient during development. For a production sample, use a validated strict `max_weight` when possible. `grow_envelope=True` avoids failure when the pilot misses a larger weight, but the report must still be checked and the resulting procedure documented.

Keep generated and accepted MC distinct. Run generated events through detector simulation and the complete selection before using them as the normalization sample of a data fit.

