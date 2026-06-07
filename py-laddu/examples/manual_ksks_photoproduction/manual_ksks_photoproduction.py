"""Manual construction sketch for gamma p -> X p, X -> K_S K_S.

The channel supplies validated JLS couplings for the X decay. The model is then
assembled with ordinary Python loops over couplings, spin projections, helicity
sectors, dynamics, angular factors, and coefficients.
"""

from fractions import Fraction

import laddu as ld


def main() -> None:
    channel = ld.Channel()
    channel.create_production('production', ['gamma', 'target'], ['X', 'recoil'])
    channel.create_decay('x_decay', 'X', ['Ks1', 'Ks2'], rules='strong')

    channel.edit_particle(
        'gamma',
        source=ld.ParticleSource.Stored,
        properties=ld.ParticleProperties(spin=1, parity='-'),
        mass=0.0,
    )
    channel.edit_particle(
        'target',
        source=ld.ParticleSource.Missing,
        properties=ld.ParticleProperties(spin=Fraction(1, 2), parity='+'),
        mass=0.938,
    )
    channel.edit_particle(
        'recoil',
        source=ld.ParticleSource.Stored,
        properties=ld.ParticleProperties(spin=Fraction(1, 2), parity='+'),
        mass=0.938,
    )

    kshort = ld.ParticleProperties(
        species='K_S',
        self_conjugate=True,
        spin=0,
        parity='-',
        strangeness=0,
        baryon_number=0,
        statistics=ld.Statistics.Boson,
    )
    channel.edit_particle(
        'Ks1', source=ld.ParticleSource.Stored, properties=kshort, mass=0.498
    )
    channel.edit_particle(
        'Ks2', source=ld.ParticleSource.Stored, properties=kshort, mass=0.498
    )

    production_frame = ld.Frame(
        'production',
        ld.Axes.from_y_z(
            ld.Axis.normal('gamma', 'recoil').at('production'),
            ld.Axis.particle('gamma').at('production'),
        ),
    )
    decay_frame = ld.Frame(
        'x_decay',
        ld.Axes.from_y_z(
            ld.Axis.normal('gamma', 'recoil').at('production').flipped(),
            ld.Axis.opposite('recoil').at('production'),
        ),
    )
    production_angles = channel.angles('X', production_frame)
    decay_angles = channel.angles('Ks1', decay_frame)
    polarization = channel.polarization(
        'production', pol_magnitude='pol_magnitude', pol_angle='pol_angle'
    )
    x_mass = channel.mass('X')

    proton_sectors = [
        ('nonflip', Fraction(1, 2), Fraction(1, 2)),
        ('flip', Fraction(1, 2), Fraction(-1, 2)),
    ]
    photon_helicities = [-1, 1]

    intensity = ld.Zero()
    couplings = channel.two_body_couplings('x_decay', j_max=2, l_max=2)
    for sector, target_helicity, recoil_helicity in proton_sectors:
        for photon_helicity in photon_helicities:
            amp = helicity_amplitude(
                couplings,
                production_angles,
                decay_angles,
                x_mass,
                sector,
                target_helicity,
                recoil_helicity,
                photon_helicity,
            )
            for photon_helicity_prime in photon_helicities:
                amp_prime = helicity_amplitude(
                    couplings,
                    production_angles,
                    decay_angles,
                    x_mass,
                    sector,
                    target_helicity,
                    recoil_helicity,
                    photon_helicity_prime,
                )
                rho = ld.PhotonSDME(
                    f'rho_{photon_helicity}_{photon_helicity_prime}',
                    helicity=photon_helicity,
                    helicity_prime=photon_helicity_prime,
                    polarization=polarization,
                )
                intensity = intensity + (rho * amp * amp_prime.conj()).real()

    print(
        f'built {len(couplings)} validated X -> K_S K_S couplings '
        f'and {intensity.n_parameters} parameters'
    )


def helicity_amplitude(
    couplings: list[ld.TwoBodyCoupling],
    production_angles: ld.Angles,
    decay_angles: ld.Angles,
    x_mass: ld.Mass,
    sector: str,
    target_helicity: int | Fraction,
    recoil_helicity: int | Fraction,
    photon_helicity: int,
) -> ld.Expression:
    coherent_sum = ld.Zero()
    for coupling in couplings:
        daughter_spin_cg = ld.clebsch_gordan(0, 0, 0, 0, 0, 0)
        for x_projection in ld.allowed_projections(coupling.j):
            production = ld.WignerD(
                (
                    f'D_prod_{coupling.wave}_{sector}'
                    f'_h{photon_helicity}_{projection_label(x_projection)}'
                ),
                spin=coupling.j,
                row_projection=x_projection,
                column_projection=0,
                angles=production_angles,
            )
            decay = ld.WignerD(
                f'D_decay_{coupling.wave}_{projection_label(x_projection)}',
                spin=coupling.j,
                row_projection=x_projection,
                column_projection=0,
                angles=decay_angles,
            )
            for family, dynamics in dynamics_for(coupling, x_mass):
                coefficient = coefficient_for(
                    family,
                    coupling.wave.label,
                    sector,
                    target_helicity,
                    recoil_helicity,
                    photon_helicity,
                    x_projection,
                )
                coherent_sum = (
                    coherent_sum
                    + daughter_spin_cg * coefficient * dynamics * production * decay
                )
    return coherent_sum


def dynamics_for(
    coupling: ld.TwoBodyCoupling, mass: ld.Mass
) -> list[tuple[str, ld.Expression]]:
    if coupling.j == 0:
        return [
            (
                'f0',
                ld.KopfKMatrixF0(
                    'f0',
                    couplings=kmatrix_couplings('f0', 5),
                    channel=ld.KopfKMatrixF0Channel.KKbar,
                    mass=mass,
                ),
            ),
            (
                'a0',
                ld.KopfKMatrixA0(
                    'a0',
                    couplings=kmatrix_couplings('a0', 2),
                    channel=ld.KopfKMatrixA0Channel.KKbar,
                    mass=mass,
                ),
            ),
        ]
    if coupling.j == 2 and coupling.l == 2:
        return [
            (
                'f2',
                ld.KopfKMatrixF2(
                    'f2',
                    couplings=kmatrix_couplings('f2', 4),
                    channel=ld.KopfKMatrixF2Channel.KKbar,
                    mass=mass,
                ),
            ),
            (
                'a2',
                ld.KopfKMatrixA2(
                    'a2',
                    couplings=kmatrix_couplings('a2', 2),
                    channel=ld.KopfKMatrixA2Channel.KKbar,
                    mass=mass,
                ),
            ),
        ]
    return []


def kmatrix_couplings(
    prefix: str, count: int
) -> tuple[tuple[ld.Parameter, ld.Parameter], ...]:
    return tuple(
        (
            ld.parameter(f'{prefix}_pole_{i}_re', initial=0.1),
            ld.parameter(f'{prefix}_pole_{i}_im', initial=0.0),
        )
        for i in range(count)
    )


def coefficient_for(
    family: str,
    wave: str,
    sector: str,
    target_helicity: int | Fraction,
    recoil_helicity: int | Fraction,
    photon_helicity: int,
    x_projection: int | Fraction,
) -> ld.Expression:
    name = (
        f'C_{family}_{wave}_{sector}'
        f'_t{projection_label(target_helicity)}'
        f'_r{projection_label(recoil_helicity)}'
        f'_h{photon_helicity}'
        f'_x{projection_label(x_projection)}'
    )
    return ld.ComplexScalar(
        name,
        'coefficient',
        re=ld.parameter(f'{name}_re', initial=1.0),
        im=ld.parameter(f'{name}_im', initial=0.0),
    )


def projection_label(projection: int | Fraction) -> str:
    doubled = 2 * projection
    value = doubled.numerator if isinstance(doubled, Fraction) else doubled
    return f'm{abs(value)}' if value < 0 else f'p{value}'


if __name__ == '__main__':
    main()
