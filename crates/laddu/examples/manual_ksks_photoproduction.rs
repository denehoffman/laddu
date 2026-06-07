//! Manual construction sketch for linearly-polarized `gamma p -> X p`, `X -> K_S K_S`.
//!
//! This example is intentionally explicit. The channel supplies validated `JLS`
//! couplings for the `X -> K_S K_S` decay, but the expression is assembled by
//! ordinary loops over couplings, spin projections, helicity sectors, dynamics,
//! angular factors, and coefficients.

use std::array;

use laddu::{
    amplitudes::{
        angular::{PhotonHelicity, PhotonPolarization, PhotonSDME, WignerD},
        kmatrix::{
            KopfKMatrixA0, KopfKMatrixA0Channel, KopfKMatrixA2, KopfKMatrixA2Channel,
            KopfKMatrixF0, KopfKMatrixF0Channel, KopfKMatrixF2, KopfKMatrixF2Channel,
        },
        scalar::ComplexScalar,
    },
    j, m,
    math::clebsch_gordan,
    parameter,
    reaction::TwoBodyCoupling,
    Axes, Axis, Channel, Expression, Frame, LadduResult, Parameter, Parity, ParticleProperties,
    RuleSet, Statistics, J, L, M,
};

fn main() -> LadduResult<()> {
    let mut channel = Channel::new();
    channel.create_production("production", ["gamma", "target"], ["X", "recoil"])?;
    channel
        .create_decay("x_decay", "X", ["Ks1", "Ks2"])?
        .rules(RuleSet::strong());

    channel
        .edit_particle("gamma")?
        .stored()
        .properties(ParticleProperties::jp(J::int(1), Parity::Negative).with_mass(0.0));
    channel
        .edit_particle("target")?
        .missing()?
        .properties(ParticleProperties::jp(J::half(1), Parity::Positive).with_mass(0.938));
    channel
        .edit_particle("recoil")?
        .stored()
        .properties(ParticleProperties::jp(J::half(1), Parity::Positive).with_mass(0.938));

    let kshort = ParticleProperties::jp(J::int(0), Parity::Negative)
        .with_species("K_S")
        .with_self_conjugate(true)
        .with_strangeness(0)
        .with_baryon_number(0)
        .with_statistics(Statistics::Boson)?
        .with_mass(0.498);
    channel
        .edit_particle("Ks1")?
        .stored()
        .properties(kshort.clone());
    channel.edit_particle("Ks2")?.stored().properties(kshort);

    let production_frame = Frame::new(
        "production",
        Axes::from_y_z(
            Axis::normal("gamma", "recoil").at("production"),
            Axis::particle("gamma").at("production"),
        ),
    )?;
    let decay_frame = Frame::new(
        "x_decay",
        Axes::from_y_z(
            Axis::normal("gamma", "recoil").at("production").flipped(),
            Axis::opposite("recoil").at("production"),
        ),
    )?;
    let production_angles = channel.angles("X", production_frame)?;
    let decay_angles = channel.angles("Ks1", decay_frame)?;
    let polarization = channel.polarization("production", "pol_magnitude", "pol_angle")?;
    let x_mass = channel.mass("X")?;

    let proton_sectors = [
        ("nonflip", M::half(1), M::half(1)),
        ("flip", M::half(1), M::half(-1)),
    ];
    let photon_helicities = [-1, 1];

    let mut intensity = Expression::zero();
    let couplings = channel.two_body_couplings("x_decay", J::int(2), L::int(2))?;
    for (sector, target_helicity, recoil_helicity) in proton_sectors {
        for photon_helicity in photon_helicities {
            let amp = helicity_amplitude(
                &couplings,
                &production_angles,
                &decay_angles,
                &x_mass,
                sector,
                target_helicity,
                recoil_helicity,
                photon_helicity,
            )?;
            for photon_helicity_prime in photon_helicities {
                let amp_prime = helicity_amplitude(
                    &couplings,
                    &production_angles,
                    &decay_angles,
                    &x_mass,
                    sector,
                    target_helicity,
                    recoil_helicity,
                    photon_helicity_prime,
                )?;
                let rho = PhotonSDME::new(
                    format!("rho_{}_{}", photon_helicity, photon_helicity_prime),
                    PhotonPolarization::Linear(Box::new(polarization.clone())),
                    PhotonHelicity::new(photon_helicity)?,
                    PhotonHelicity::new(photon_helicity_prime)?,
                )?;
                intensity += (rho * amp.clone() * amp_prime.conj()).real();
            }
        }
    }

    println!(
        "built {} validated X -> K_S K_S couplings and {} parameters",
        couplings.len(),
        intensity.n_parameters()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn helicity_amplitude(
    couplings: &[TwoBodyCoupling],
    production_angles: &laddu::Angles,
    decay_angles: &laddu::Angles,
    x_mass: &laddu::Mass,
    sector: &str,
    target_helicity: M,
    recoil_helicity: M,
    photon_helicity: i32,
) -> LadduResult<Expression> {
    let mut coherent_sum = Expression::zero();
    for coupling in couplings {
        let daughter_spin_cg = clebsch_gordan(j!(0), m!(0), j!(0), m!(0), j!(0), m!(0));
        for x_projection in coupling.j().projections() {
            let production = WignerD::new(
                format!(
                    "D_prod_{}_{}_h{}_{}",
                    coupling.wave(),
                    sector,
                    photon_helicity,
                    projection_label(x_projection)
                ),
                coupling.j(),
                x_projection,
                M::int(0),
                production_angles,
            )?;
            let decay = WignerD::new(
                format!(
                    "D_decay_{}_{}",
                    coupling.wave(),
                    projection_label(x_projection)
                ),
                coupling.j(),
                x_projection,
                M::int(0),
                decay_angles,
            )?;

            for (family, dynamics) in dynamics_for(coupling.j(), coupling.l(), x_mass)? {
                let coefficient = coefficient_for(
                    &family,
                    coupling.wave().label(),
                    sector,
                    target_helicity,
                    recoil_helicity,
                    photon_helicity,
                    x_projection,
                )?;
                coherent_sum +=
                    daughter_spin_cg * coefficient * dynamics * production.clone() * decay.clone();
            }
        }
    }
    Ok(coherent_sum)
}

fn dynamics_for(j: J, l: L, mass: &laddu::Mass) -> LadduResult<Vec<(String, Expression)>> {
    match j.doubled() {
        0 => Ok(vec![
            (
                "f0".to_string(),
                KopfKMatrixF0::new(
                    "f0",
                    kmatrix_couplings("f0"),
                    KopfKMatrixF0Channel::KKbar,
                    mass,
                    None,
                )?,
            ),
            (
                "a0".to_string(),
                KopfKMatrixA0::new(
                    "a0",
                    kmatrix_couplings("a0"),
                    KopfKMatrixA0Channel::KKbar,
                    mass,
                    None,
                )?,
            ),
        ]),
        4 if l == L::int(2) => Ok(vec![
            (
                "f2".to_string(),
                KopfKMatrixF2::new(
                    "f2",
                    kmatrix_couplings("f2"),
                    KopfKMatrixF2Channel::KKbar,
                    mass,
                    None,
                )?,
            ),
            (
                "a2".to_string(),
                KopfKMatrixA2::new(
                    "a2",
                    kmatrix_couplings("a2"),
                    KopfKMatrixA2Channel::KKbar,
                    mass,
                    None,
                )?,
            ),
        ]),
        _ => Ok(Vec::new()),
    }
}

fn kmatrix_couplings<const N: usize>(prefix: &str) -> [[Parameter; 2]; N] {
    array::from_fn(|i| {
        [
            parameter!(format!("{prefix}_pole_{i}_re"), initial: 0.1),
            parameter!(format!("{prefix}_pole_{i}_im"), initial: 0.0),
        ]
    })
}

fn coefficient_for(
    family: &str,
    wave: String,
    sector: &str,
    target_helicity: M,
    recoil_helicity: M,
    photon_helicity: i32,
    x_projection: M,
) -> LadduResult<Expression> {
    let name = format!(
        "C_{family}_{wave}_{sector}_t{}_r{}_h{}_x{}",
        projection_label(target_helicity),
        projection_label(recoil_helicity),
        photon_helicity,
        projection_label(x_projection),
    );
    ComplexScalar::new(
        [&name, "coefficient"],
        parameter!(format!("{name}_re"), initial: 1.0),
        parameter!(format!("{name}_im"), initial: 0.0),
    )
}

fn projection_label(projection: M) -> String {
    match projection.doubled() {
        value if value < 0 => format!("m{}", value.abs()),
        value => format!("p{value}"),
    }
}
