# Building a polarized photoproduction model

This chapter applies the expression tools to a sequential helicity amplitude
for $\gamma p\to Xp$, followed by $X\to ab$. The purpose is the construction
pattern: an analysis must still document and validate its own phase, frame, and
polarization conventions.

## Sequential helicity amplitude

For photon, target, recoil, and resonance helicities
$\lambda_\gamma,\lambda_t,\lambda_r,\lambda_X$, a typical term is

$$
\mathcal A_{\lambda_\gamma\lambda_t\lambda_r}
=\mathcal R(s)\sum_{\lambda_X}
C_\mathrm{prod}
D^{J_P*}_{\lambda_\gamma-\lambda_t,\lambda_X-\lambda_r}
(\phi_P,\theta_P,0)
C_\mathrm{dec}
D^{J_X*}_{\lambda_X,\lambda_a-\lambda_b}
(\phi_H,\theta_H,0).
$$

Build the production and decay angles from named vertices:

```python
production = generation_channel.vertex("production")
decay = generation_channel.vertex("decay")

beam_axis = production.vec3("gamma")
helicity_axis = production.vec3("X")
production_normal = beam_axis.cross(helicity_axis)

theta_p = production.theta("X", z_axis=beam_axis, y_hint=ld.Vec3.y_axis())
phi_p = production.phi("X", z_axis=beam_axis, y_hint=ld.Vec3.y_axis())
theta_h = decay.theta("ks1", z_axis=helicity_axis, y_hint=production_normal)
phi_h = decay.phi("ks1", z_axis=helicity_axis, y_hint=production_normal)
```

For one helicity combination, typed quantum numbers enter Wigner functions and
Clebsch–Gordan coefficients directly:

```python
d_production = ld.WignerD(J_production, m_initial, m_final).D(
    alpha=phi_p, beta=theta_p
).conj()
d_decay = ld.WignerD(J_x, lambda_x, lambda_decay).D(
    alpha=phi_h, beta=theta_h
).conj()

helicity_term = line_shape * cg_production * d_production * cg_decay * d_decay
```

The discrete values `J_production`, `m_initial`, and the Clebsch–Gordan factors
come from the wave hypothesis developed in {doc}`quantum-numbers` and
{doc}`quantum-rules`. Sum amplitudes coherently over indistinguishable states
and intensities incoherently over unobserved orthogonal states.

## Linear beam polarization

In the photon-helicity basis,

$$
\rho^\gamma=\frac12
\begin{pmatrix}
1 & -P_\gamma e^{-2i\Phi}\\
-P_\gamma e^{2i\Phi} & 1
\end{pmatrix}.
$$

If $\mathcal A_+$ and $\mathcal A_-$ are the two photon-helicity amplitudes,
the intensity is $\boldsymbol{\mathcal A}^\dagger\rho^\gamma
\boldsymbol{\mathcal A}$:

```python
polarization = ld.scalar("beam_polarization")
polarization_angle = ld.scalar("polarization_angle")

rho_gamma = 0.5 * ld.matrix(
    [
        [1.0, -polarization * ld.cis(-2.0 * polarization_angle)],
        [-polarization * ld.cis(2.0 * polarization_angle), 1.0],
    ]
)
helicity_amplitudes = ld.vector([amplitude_plus, amplitude_minus])
intensity = helicity_amplitudes.conj() @ (
    rho_gamma @ helicity_amplitudes
)
model = ld.Model(intensity)
```

This contraction preserves interference between photon helicities. Adding
their intensities would discard the off-diagonal polarization information.

Validate parity relations, rotations of the polarization plane, the
$P_\gamma\to0$ limit, positivity over generated MC, and recovery of injected
parameters in pseudo-data before applying the model to observations.
