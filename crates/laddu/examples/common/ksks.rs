//! Shared gamma p -> X p, X -> K_S K_S channel and amplitude model.

use std::f64::consts::PI;

use laddu::prelude::*;

pub const F2_MAGNITUDE_TRUTH: f64 = 0.65;
pub const F2_PHASE_TRUTH: f64 = 0.7;

// PDG 2025 values queried with the local `pdg` CLI.
const F0_MASS: f64 = 1.522;
const F0_WIDTH: f64 = 0.108;
const F2_MASS: f64 = 1.275_412_049_919_005;
const F2_WIDTH: f64 = 0.186_554_356_637_326_4;

pub fn ksks_channel() -> LadduPhysicsResult<Channel> {
    let mut channel = Channel::new("gamma p -> K_S K_S p");
    let k_short_mass = particles::K_SHORT.mass()?;
    channel
        .edge("gamma")
        .p4(Vec4::event("gamma"))
        .properties(&particles::PHOTON)
        .initial_energy_source_direction(ScalarSource::uniform(8.0, 9.0), RealVec3::z())
        .output();
    channel
        .edge("target")
        .p4(Vec4::event("target"))
        .properties(&particles::PROTON)
        .initial_momentum(RealVec3::default())
        .output();
    channel
        .edge("X")
        .mass_proposal(MassProposal::uniform(2.0 * k_short_mass, 2.0))
        .generated_only();
    channel
        .edge("recoil")
        .p4(Vec4::event("recoil"))
        .properties(&particles::PROTON)
        .output();
    channel
        .edge("ks1")
        .p4(Vec4::event("ks1"))
        .properties(&particles::K_SHORT)
        .output();
    channel
        .edge("ks2")
        .p4(Vec4::event("ks2"))
        .properties(&particles::K_SHORT)
        .output();

    channel
        .vertex("production")
        .incoming(["gamma", "target"])
        .outgoing(["X", "recoil"])
        .generation(VertexProposal::t_exchange(
            ("gamma", "X"),
            // The exponential component efficiently covers forward t exchange;
            // the uniform defensive component keeps 1/q(t) bounded everywhere.
            TDistribution::mixture([
                (0.2, TComponent::Uniform),
                (0.8, TComponent::Exponential { slope: 4.0 }),
            ]),
        ));
    channel
        .vertex("decay")
        .incoming(["X"])
        .outgoing(["ks1", "ks2"]);
    Ok(channel)
}

/// Unpolarized sequential-helicity intensity.
///
/// The production partial wave has total J=1/2 for f0 p and J=3/2 for f2 p. Each external
/// helicity channel is coherent in f0/f2 and the external helicities are then
/// averaged/summed incoherently. The two spin-zero kaons have S=0 and L=J;
/// only the even J=0,2 waves used here are Bose symmetric under ks1 <-> ks2.
/// Build one coherent intensity whose amplitude components can be projected by tag.
pub fn ksks_intensity(channel: &Channel) -> LadduPhysicsResult<Expr> {
    let s = channel.s("X")?;
    let k_short_mass = channel.particle("ks1")?.mass()?;
    let f0 = ParticleProperties::unknown()
        .with_name("f0(1500)")
        .with_spin(j!(0))
        .with_parity(Parity::Positive)
        .with_mass(F0_MASS);
    let f2 = ParticleProperties::unknown()
        .with_name("f2(1270)")
        .with_spin(j!(2))
        .with_parity(Parity::Positive)
        .with_mass(F2_MASS);
    let f0_bw =
        relativistic_breit_wigner(&s, f0.mass()?, F0_WIDTH, k_short_mass, k_short_mass, l!(0))?;
    let f2_bw =
        relativistic_breit_wigner(&s, f2.mass()?, F2_WIDTH, k_short_mass, k_short_mass, l!(2))?;
    let photon_helicities = channel
        .particle("gamma")?
        .spin()?
        .projections()
        .into_iter()
        .filter(|projection| *projection != M::int(0))
        .collect::<Vec<_>>();
    let target_helicities = channel.particle("target")?.spin()?.projections();
    let recoil_helicities = channel.particle("recoil")?.spin()?.projections();
    let first_kaon_helicities = channel.particle("ks1")?.spin()?.projections();
    let second_kaon_helicities = channel.particle("ks2")?.spin()?.projections();
    let f0_decay_wave = unique_decay_partial_wave(channel, &f0)?;
    let f2_decay_wave = unique_decay_partial_wave(channel, &f2)?;
    let f2_coupling = polar_complex(
        parameter!(
            "f2_magnitude",
            initial: 0.35,
            bounds: (0.0, 2.0),
            scale: 0.5
        ),
        parameter!(
            "f2_phase",
            initial: 0.0,
            bounds: (-PI, PI),
            periodic: true,
            scale: 1.0
        ),
    );
    let mut coherent = Expr::from(0.0);
    for photon in photon_helicities {
        for &target in &target_helicities {
            for &recoil in &recoil_helicities {
                for &first_kaon in &first_kaon_helicities {
                    for &second_kaon in &second_kaon_helicities {
                        let f0 = sequential_wave(
                            channel,
                            &f0,
                            f0_decay_wave,
                            photon,
                            target,
                            recoil,
                            first_kaon,
                            second_kaon,
                            &f0_bw,
                        )?
                        .tagged("f0");
                        let f2 = (f2_coupling.clone()
                            * sequential_wave(
                                channel,
                                &f2,
                                f2_decay_wave,
                                photon,
                                target,
                                recoil,
                                first_kaon,
                                second_kaon,
                                &f2_bw,
                            )?)
                        .tagged("f2");
                        // The f0 coupling is the fixed scale and phase reference.
                        coherent += (f0 + f2).norm_sqr();
                    }
                }
            }
        }
    }
    Ok(coherent * 0.25)
}

#[allow(clippy::too_many_arguments)]
fn sequential_wave(
    channel: &Channel,
    resonance: &ParticleProperties,
    decay_wave: PartialWave,
    m_photon: M,
    m_target: M,
    m_recoil: M,
    m_ks1: M,
    m_ks2: M,
    line_shape: &Expr,
) -> LadduPhysicsResult<Expr> {
    let production = channel.get_vertex("production")?;
    let decay = channel.get_vertex("decay")?;
    let beam_axis = production.vec3("gamma")?;
    let helicity_axis = production.vec3("X")?;
    let production_normal = beam_axis.cross(&helicity_axis);
    let production_theta = production.theta("X", beam_axis.clone(), Vec3::y())?;
    let production_phi = production.phi("X", beam_axis, Vec3::y())?;
    let decay_theta = decay.theta("ks1", helicity_axis.clone(), production_normal.clone())?;
    let decay_phi = decay.phi("ks1", helicity_axis, production_normal)?;
    let resonance_spin = resonance.spin()?;
    let photon_spin = channel.particle("gamma")?.spin()?;
    let target_spin = channel.particle("target")?.spin()?;
    let recoil_spin = channel.particle("recoil")?.spin()?;
    let first_kaon_spin = channel.particle("ks1")?.spin()?;
    let second_kaon_spin = channel.particle("ks2")?.spin()?;
    let production_total_j = production_total_j(channel, resonance)?;
    let decay_helicity = m_ks1 - m_ks2;
    // The target and recoil travel opposite the corresponding +z helicity axes.
    let initial_projection = m_photon - m_target;
    let initial_coupling = clebsch_gordan(
        photon_spin,
        m_photon,
        target_spin,
        -m_target,
        production_total_j,
        initial_projection,
    );
    let mut angular = Expr::from(0.0);
    for resonance_helicity in resonance_spin.projections() {
        let final_projection = resonance_helicity - m_recoil;
        let production_coupling = clebsch_gordan(
            resonance_spin,
            resonance_helicity,
            recoil_spin,
            -m_recoil,
            production_total_j,
            final_projection,
        );
        let daughter_spin_coupling = clebsch_gordan(
            first_kaon_spin,
            m_ks1,
            second_kaon_spin,
            -m_ks2,
            decay_wave.s,
            decay_helicity,
        );
        let orbital_coupling = clebsch_gordan(
            J::from(decay_wave.l),
            m!(0),
            decay_wave.s,
            decay_helicity,
            decay_wave.j,
            decay_helicity,
        );
        if initial_coupling == 0.0
            || production_coupling == 0.0
            || daughter_spin_coupling == 0.0
            || orbital_coupling == 0.0
        {
            continue;
        }
        let production_d =
            WignerDMatrix::new(production_total_j, initial_projection, final_projection)?
                .D(&production_phi, &production_theta, 0.0)
                .conj();
        let decay_d = WignerDMatrix::new(resonance_spin, resonance_helicity, decay_helicity)?
            .D(&decay_phi, &decay_theta, 0.0)
            .conj();
        angular += initial_coupling
            * production_coupling
            * daughter_spin_coupling
            * orbital_coupling
            * production_d
            * decay_d;
    }
    let normalization = (f64::from(production_total_j.multiplicity())
        * f64::from(resonance_spin.multiplicity()))
    .sqrt()
        / (4.0 * PI);
    let ls_normalization =
        (f64::from(decay_wave.l.multiplicity()) / f64::from(decay_wave.j.multiplicity())).sqrt();
    Ok(normalization * ls_normalization * line_shape * angular)
}

fn unique_decay_partial_wave(
    channel: &Channel,
    resonance: &ParticleProperties,
) -> LadduPhysicsResult<PartialWave> {
    let resonance_spin = resonance.spin()?;
    let first_daughter = channel.particle("ks1")?;
    let second_daughter = channel.particle("ks2")?;
    let max_l = L::try_from(resonance_spin + first_daughter.spin()? + second_daughter.spin()?)?;
    Ok(SelectionRules::angular(max_l)
        .allowed_partial_waves(resonance, (first_daughter, second_daughter))
        .into_iter()
        .next()
        .map(|allowed| allowed.wave)
        .expect("there should be at least one valid wave"))
}

fn production_total_j(channel: &Channel, resonance: &ParticleProperties) -> LadduPhysicsResult<J> {
    let initial = SelectionRules::coupled_spins(
        channel.particle("gamma")?.spin()?,
        channel.particle("target")?.spin()?,
    );
    let final_state =
        SelectionRules::coupled_spins(resonance.spin()?, channel.particle("recoil")?.spin()?);
    Ok(initial
        .into_iter()
        .find(|candidate| final_state.contains(candidate))
        .expect("there should be at least one valid coupling"))
}

pub fn truth_parameters(model: &CompiledModel) -> Result<ParamValues, ParamError> {
    let free = model
        .params()
        .free_values_with(|parameter| match parameter.name() {
            "f2_magnitude" => F2_MAGNITUDE_TRUTH,
            "f2_phase" => F2_PHASE_TRUTH,
            name => panic!("unexpected free parameter `{name}` in K_S K_S model"),
        });
    model.params().values(&free)
}

pub fn print_report(label: &str, report: &GenerationReport) {
    println!(
        "{label}: {} events from {} proposals (acceptance {:.2}%, max weight {:.3e})",
        report.produced,
        report.proposals,
        100.0 * report.acceptance_rate(),
        report.maximum_weight,
    );
}
