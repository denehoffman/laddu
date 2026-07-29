# A complex model: polarized photoproduction

This tutorial outlines a helicity construction for $\gamma p\to Xp$, followed by $X\to ab$. It is intentionally explicit: for a production model, audit phase conventions, frames, and quantum-number selections against the collaboration convention.

## Sequential helicity amplitude

For photon helicity $\lambda_\gamma$, target helicity $\lambda_t$, recoil helicity $\lambda_r$, and resonance helicity $\lambda_X$, a typical sequential amplitude contains

$$
\mathcal A_{\lambda_\gamma\lambda_t\lambda_r}
=\mathcal R(s)\sum_{\lambda_X}
C_\mathrm{prod}\,
D^{J_P*}_{\lambda_\gamma-\lambda_t,\lambda_X-\lambda_r}
(\phi_P,\theta_P,0)\,
C_\mathrm{dec}\,
D^{J_X*}_{\lambda_X,\lambda_a-\lambda_b}
(\phi_H,\theta_H,0).
$$

The complete, executable construction of axes, Clebsch–Gordan factors, Wigner functions, and relativistic $S$- and $D$-wave line shapes is provided in the {doc}`closure` tutorial. The essential laddu pattern is:

```python
production = channel.vertex("production")
decay = channel.vertex("decay")
beam_axis = production.vec3("gamma")
helicity_axis = production.vec3("X")
production_normal = beam_axis.cross(helicity_axis)

theta_p = production.theta("X", beam_axis, ld.Vec3.y_axis())
phi_p = production.phi("X", beam_axis, ld.Vec3.y_axis())
theta_h = decay.theta("a", helicity_axis, production_normal)
phi_h = decay.phi("a", helicity_axis, production_normal)

d_prod = ld.WignerD(J_prod, m_initial, m_final).D(phi_p, theta_p).conj()
d_decay = ld.WignerD(J_x, lambda_x, lambda_decay).D(phi_h, theta_h).conj()
wave = line_shape * coupling_factors * d_prod * d_decay
```

## Linear beam polarization

The photon density matrix in the helicity basis may be written

$$
\rho^\gamma=\frac12
\begin{pmatrix}
1 & -P_\gamma e^{-2i\Phi}\\
-P_\gamma e^{2i\Phi} & 1
\end{pmatrix},
$$

where $P_\gamma$ and $\Phi$ are event-by-event scalar columns. If `a_plus` and
`a_minus` are the amplitudes for photon helicities $+1$ and $-1$, contract the
density matrix rather than adding the two intensities incoherently:

```python
p_gamma = ld.scalar("beam_polarization")
phi = ld.scalar("polarization_angle")

rho = 0.5 * ld.matrix(
    [
        [1.0, -p_gamma * ld.cis(-2.0 * phi)],
        [-p_gamma * ld.cis(2.0 * phi), 1.0],
    ]
)
amplitudes = ld.vector([a_plus, a_minus])
intensity = amplitudes.conj() @ (rho @ amplitudes)
model = ld.Model(intensity)
```

This is the expression-level form of
$\boldsymbol{\mathcal A}^\dagger\rho^\gamma\boldsymbol{\mathcal A}$.

Sum coherently over amplitudes that lead to the same observed quantum state, and incoherently over unobserved orthogonal helicities. A reliable implementation makes those two operations visually distinct in the code.

## Production couplings and tags

```python
rho_mag = ld.parameter("rho_mag", initial=0.2, bounds=(0.0, 2.0))
rho_phase = ld.parameter("rho_phase", initial=0.0, bounds=(-3.14159, 3.14159), periodic=True)
rho = ld.polar_complex(rho_mag, rho_phase)

coherent = reference_wave.tagged("reference") + rho * second_wave.tagged("second")
```

Before fitting data, test parity relations, rotations of the polarization plane, the $P_\gamma\to0$ limit, positivity on a large MC sample, and recovery of injected couplings in closure tests.
