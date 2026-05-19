use crate::{
    quantum::{PartialWave, ParticleProperties},
    AllowedPartialWave, AngularMomentum, OrbitalAngularMomentum, Parity, Statistics,
};

/// A collection of optional selection rules for testing whether a two-body
/// decay channel is allowed.
///
/// Each boolean enables one conservation or symmetry check. Disabled checks are
/// ignored. Enabled checks are treated permissively with respect to missing
/// quantum numbers: if the required values are not known, that check does not
/// reject the channel.
///
/// The purely angular-momentum constraints are not represented here. Those are
/// handled separately when constructing candidate partial waves.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Default)]
pub struct RuleSet {
    /// Enforce intrinsic parity conservation.
    ///
    /// For a two-body final state this checks
    /// $`P_\text{parent} = P_a P_b (-1)^L`$.
    pub parity: bool,

    /// Enforce total isospin coupling.
    ///
    /// This checks whether the two daughter isospins can couple to the parent
    /// isospin:
    /// $`I_\text{parent} \in |I_a - I_b|, \ldots, I_a + I_b`$.
    pub isospin: bool,

    /// Enforce conservation of the isospin projection $`I_3`$.
    ///
    /// This checks $`I_{3,\text{parent}} = I_{3,a} + I_{3,b}`$.
    pub isospin_projection: bool,

    /// Enforce charge-conjugation parity conservation when applicable.
    ///
    /// This is only meaningful for states with a defined $`C`$ eigenvalue and
    /// final states that can be interpreted as $`C`$ eigenstates, such as
    /// suitable particle-antiparticle combinations.
    pub c_parity: bool,

    /// Enforce G-parity conservation when applicable.
    ///
    /// This is mainly useful for light-quark isospin multiplets where
    /// $`G`$-parity is defined. It should not be enabled blindly for arbitrary
    /// hadrons.
    pub g_parity: bool,

    /// Enforce electric charge conservation.
    ///
    /// This checks $`Q_\text{parent} = Q_a + Q_b`$.
    pub charge: bool,

    /// Enforce strangeness conservation.
    ///
    /// This checks $`S_\text{parent} = S_a + S_b`$.
    ///
    /// Strong and electromagnetic interactions conserve strangeness; weak
    /// interactions generally do not.
    pub strangeness: bool,

    /// Enforce charm conservation.
    ///
    /// This checks $`C_\text{parent} = C_a + C_b`$, where $`C`$ here denotes
    /// charm quantum number, not charge conjugation.
    pub charm: bool,

    /// Enforce bottomness conservation.
    ///
    /// This checks $`B'_\text{parent} = B'_a + B'_b`$, where $`B'`$ denotes
    /// bottomness, not baryon number.
    pub bottomness: bool,

    /// Enforce topness conservation.
    ///
    /// This checks $`T_\text{parent} = T_a + T_b`$.
    pub topness: bool,

    /// Enforce baryon-number conservation.
    ///
    /// This checks $`B_\text{parent} = B_a + B_b`$.
    pub baryon_number: bool,

    /// Enforce electron-family lepton-number conservation.
    ///
    /// This checks $`L_e(\text{parent}) = L_e(a) + L_e(b)`$.
    pub electron_lepton_number: bool,

    /// Enforce muon-family lepton-number conservation.
    ///
    /// This checks $`L_\mu(\text{parent}) = L_\mu(a) + L_\mu(b)`$.
    pub muon_lepton_number: bool,

    /// Enforce tau-family lepton-number conservation.
    ///
    /// This checks $`L_\tau(\text{parent}) = L_\tau(a) + L_\tau(b)`$.
    pub tau_lepton_number: bool,

    /// Enforce total lepton-number conservation.
    ///
    /// This checks $`L_\text{parent} = L_a + L_b`$, where
    /// $`L = L_e + L_\mu + L_\tau`$.
    ///
    /// This is independent of the individual lepton-family checks. If both this
    /// and the family-specific checks are enabled, all enabled checks must pass.
    pub lepton_number: bool,

    /// Enforce exchange-symmetry constraints for identical final-state
    /// particles when enough information is available.
    ///
    /// At minimum, this is useful for cases such as identical spin-zero bosons,
    /// where only even $`L`$ is allowed.
    pub identical_particle_symmetry: bool,
}
impl RuleSet {
    /// Construct a rule set with no non-angular selection rules enabled.
    ///
    /// This is useful when only the angular-momentum coupling constraints should
    /// be applied:
    /// $`S \in |j_a - j_b|, \ldots, j_a + j_b`$
    /// and
    /// $`J \in |L - S|, \ldots, L + S`$.
    pub fn angular() -> Self {
        Self::default()
    }

    /// Construct a rule set appropriate for ordinary strong two-body decays.
    ///
    /// This enables parity, isospin, isospin projection, electric charge,
    /// flavor quantum numbers, baryon number, and identical-particle exchange
    /// symmetry.
    ///
    /// Charge-conjugation parity and G-parity are left disabled because they
    /// are only meaningful for certain channels and should be enabled
    /// explicitly when applicable.
    pub fn strong() -> Self {
        Self {
            parity: true,
            isospin: true,
            isospin_projection: true,
            charge: true,
            strangeness: true,
            charm: true,
            bottomness: true,
            topness: true,
            baryon_number: true,
            identical_particle_symmetry: true,
            ..Default::default()
        }
    }

    /// Construct a rule set appropriate for electromagnetic two-body decays.
    ///
    /// This enables parity, electric charge, flavor quantum numbers, baryon
    /// number, isospin-projection conservation, and identical-particle exchange
    /// symmetry.
    ///
    /// Total isospin is not enabled because electromagnetic interactions break
    /// isospin symmetry.
    pub fn electromagnetic() -> Self {
        Self {
            parity: true,
            isospin_projection: true,
            charge: true,
            strangeness: true,
            charm: true,
            bottomness: true,
            topness: true,
            baryon_number: true,
            identical_particle_symmetry: true,
            ..Default::default()
        }
    }

    /// Construct a rule set appropriate for weak two-body decays.
    ///
    /// This enables electric charge, baryon number, individual lepton-family
    /// numbers, total lepton number, and identical-particle exchange symmetry.
    ///
    /// Parity, isospin, strangeness, charm, bottomness, and topness are not
    /// enabled because weak interactions can violate or change them.
    pub fn weak() -> Self {
        Self {
            charge: true,
            baryon_number: true,
            electron_lepton_number: true,
            muon_lepton_number: true,
            tau_lepton_number: true,
            lepton_number: true,
            identical_particle_symmetry: true,
            ..Default::default()
        }
    }

    /// Check whether a candidate two-body partial wave satisfies this rule set.
    ///
    /// `parent` is the decaying particle, `daughters` are the two final-state
    /// particles, `l` is their relative orbital angular momentum, and `s` is
    /// their coupled spin.
    ///
    /// Returns `false` if any enabled rule is definitely violated. Returns
    /// `true` if all enabled rules pass or if some enabled rules cannot be
    /// evaluated because the required quantum numbers are unknown.
    ///
    /// The angular-momentum coupling itself should be checked before or during
    /// candidate partial-wave construction; this method only applies the
    /// selected conservation and symmetry rules.
    pub fn check(
        &self,
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
        l: OrbitalAngularMomentum,
        s: AngularMomentum,
    ) -> bool {
        (!self.parity || Self::check_parity(parent, daughters, l).unwrap_or(true))
            && (!self.isospin || Self::check_isospin(parent, daughters).unwrap_or(true))
            && (!self.isospin_projection
                || Self::check_isospin_projection(parent, daughters).unwrap_or(true))
            && (!self.c_parity || Self::check_c_parity(parent, daughters, l, s).unwrap_or(true))
            && (!self.g_parity || Self::check_g_parity(parent, daughters).unwrap_or(true))
            && self.check_additives(parent, daughters).unwrap_or(true)
            && (!self.identical_particle_symmetry
                || Self::check_identical_particle_symmetry(daughters, l).unwrap_or(true))
    }
    fn check_parity(
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
        l: OrbitalAngularMomentum,
    ) -> Option<bool> {
        let p_parent = parent.parity?;
        let p_a = daughters.0.parity?;
        let p_b = daughters.1.parity?;
        let sign = if l.value() & 1 == 0 { 1 } else { -1 };
        Some(p_parent.value() == p_a.value() * p_b.value() * sign)
    }
    fn check_isospin(
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
    ) -> Option<bool> {
        let i_parent = parent.isospin?;
        let i_a = daughters.0.isospin?;
        let i_b = daughters.1.isospin?;
        Some(
            i_parent
                .isospin()
                .can_couple_to(i_a.isospin(), i_b.isospin()),
        )
    }
    fn check_isospin_projection(
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
    ) -> Option<bool> {
        let i_parent = parent.isospin?;
        let i_a = daughters.0.isospin?;
        let i_b = daughters.1.isospin?;
        let i3_parent = i_parent.projection()?;
        let i3_a = i_a.projection()?;
        let i3_b = i_b.projection()?;
        Some(i3_parent.value() == i3_a.value() + i3_b.value())
    }
    fn check_c_parity(
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
        l: OrbitalAngularMomentum,
        s: AngularMomentum,
    ) -> Option<bool> {
        let c_parent = parent.c_parity?;
        if !daughters.0.is_antiparticle_of(daughters.1) {
            return None;
        }
        let exp_twice = 2 * l.value() + s.value();
        if !exp_twice.is_multiple_of(2) {
            return Some(false);
        }
        let c_final = if (exp_twice / 2).is_multiple_of(2) {
            Parity::Positive
        } else {
            Parity::Negative
        };
        Some(c_parent == c_final)
    }
    fn check_g_parity(
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
    ) -> Option<bool> {
        let g_parent = parent.g_parity?;
        let g_a = daughters.0.g_parity?;
        let g_b = daughters.1.g_parity?;
        Some(g_parent.value() == g_a.value() * g_b.value())
    }
    fn check_additives(
        &self,
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
    ) -> Option<bool> {
        let mut unknown = false;

        macro_rules! check_conserved {
            ($enabled:expr, $parent:expr, $a:expr, $b:expr) => {
                if $enabled {
                    match ($parent, $a, $b) {
                        (Some(parent), Some(a), Some(b)) => {
                            if parent != a + b {
                                return Some(false);
                            }
                        }

                        _ => {
                            unknown = true;
                        }
                    }
                }
            };
        }

        check_conserved!(
            self.charge,
            parent.charge.map(|q| q.value()),
            daughters.0.charge.map(|q| q.value()),
            daughters.1.charge.map(|q| q.value())
        );

        check_conserved!(
            self.strangeness,
            parent.strangeness,
            daughters.0.strangeness,
            daughters.1.strangeness
        );

        check_conserved!(
            self.charm,
            parent.charm,
            daughters.0.charm,
            daughters.1.charm
        );

        check_conserved!(
            self.bottomness,
            parent.bottomness,
            daughters.0.bottomness,
            daughters.1.bottomness
        );

        check_conserved!(
            self.topness,
            parent.topness,
            daughters.0.topness,
            daughters.1.topness
        );

        check_conserved!(
            self.baryon_number,
            parent.baryon_number,
            daughters.0.baryon_number,
            daughters.1.baryon_number
        );

        check_conserved!(
            self.electron_lepton_number,
            parent.electron_lepton_number,
            daughters.0.electron_lepton_number,
            daughters.1.electron_lepton_number
        );

        check_conserved!(
            self.muon_lepton_number,
            parent.muon_lepton_number,
            daughters.0.muon_lepton_number,
            daughters.1.muon_lepton_number
        );

        check_conserved!(
            self.tau_lepton_number,
            parent.tau_lepton_number,
            daughters.0.tau_lepton_number,
            daughters.1.tau_lepton_number
        );

        if self.lepton_number {
            match (
                parent.electron_lepton_number,
                parent.muon_lepton_number,
                parent.tau_lepton_number,
                daughters.0.electron_lepton_number,
                daughters.0.muon_lepton_number,
                daughters.0.tau_lepton_number,
                daughters.1.electron_lepton_number,
                daughters.1.muon_lepton_number,
                daughters.1.tau_lepton_number,
            ) {
                (
                    Some(parent_e),
                    Some(parent_mu),
                    Some(parent_tau),
                    Some(a_e),
                    Some(a_mu),
                    Some(a_tau),
                    Some(b_e),
                    Some(b_mu),
                    Some(b_tau),
                ) => {
                    let parent_total = parent_e + parent_mu + parent_tau;

                    let daughter_total = a_e + a_mu + a_tau + b_e + b_mu + b_tau;

                    if parent_total != daughter_total {
                        return Some(false);
                    }
                }

                _ => {
                    unknown = true;
                }
            }
        }

        if unknown {
            None
        } else {
            Some(true)
        }
    }
    fn check_identical_particle_symmetry(
        daughters: (&ParticleProperties, &ParticleProperties),
        l: OrbitalAngularMomentum,
    ) -> Option<bool> {
        let sp_a = daughters.0.species.as_ref()?;
        let sp_b = daughters.1.species.as_ref()?;
        if sp_a != sp_b {
            return Some(true);
        }
        let stats_a = daughters.0.statistics?;
        let stats_b = daughters.0.statistics?;
        if stats_a != stats_b {
            return Some(false);
        }
        if stats_a == Statistics::Boson
            && daughters.0.spin.map(|x| x.value()) == Some(0)
            && daughters.1.spin.map(|x| x.value()) == Some(0)
        {
            if l.value().is_multiple_of(2) {
                return Some(true);
            }
            return Some(false);
        }
        None
    }
}

/// Configuration for generating and filtering allowed two-body partial waves.
///
/// `SelectionRules` combines a maximum orbital angular momentum with a
/// [`RuleSet`]. Candidate waves are generated from angular-momentum coupling
/// and are then filtered by the enabled rules.
///
/// The generated waves satisfy
/// $`S \in |j_a - j_b|, \ldots, j_a + j_b`$
/// and
/// $`J \in |L - S|, \ldots, L + S`$,
/// with $`0 \le L \le L_\text{max}`$.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SelectionRules {
    /// Maximum orbital angular momentum $`L_\text{max}`$ considered when
    /// generating candidate partial waves.
    ///
    /// The solver scans all integer values
    /// $`L = 0, 1, \ldots, L_\text{max}`$.
    pub max_l: OrbitalAngularMomentum,

    /// Conservation and symmetry rules used to filter candidate waves.
    ///
    /// Angular-momentum compatibility is handled by
    /// [`SelectionRules::allowed_partial_waves`]. The [`RuleSet`] applies
    /// additional checks such as parity, charge, isospin, flavor quantum
    /// numbers, $`C`$-parity, $`G`$-parity, and identical-particle symmetry.
    pub rules: RuleSet,
}
impl Default for SelectionRules {
    fn default() -> Self {
        Self {
            max_l: OrbitalAngularMomentum::integer(6),
            rules: RuleSet::strong(),
        }
    }
}
impl SelectionRules {
    /// Return all possible coupled total spins from two daughter spins.
    ///
    /// Given daughter spins $`j_a`$ and $`j_b`$, this returns
    /// $`S = |j_a - j_b|, |j_a - j_b| + 1, \ldots, j_a + j_b`$.
    ///
    /// Internally angular momenta are stored as doubled values, so the returned
    /// sequence advances by two in the doubled representation.
    pub fn coupled_spins(a: AngularMomentum, b: AngularMomentum) -> Vec<AngularMomentum> {
        let min = a.value().abs_diff(b.value());
        let max = a.value() + b.value();
        (min..=max)
            .step_by(2)
            .map(AngularMomentum::half_integer)
            .collect()
    }
    /// Generate all allowed two-body partial waves for a parent and two
    /// daughters.
    ///
    /// The parent spin is interpreted as the total angular momentum $`J`$ of
    /// the resonance. The daughter spins are coupled to possible total-spin
    /// values $`S`$, and each $`S`$ is combined with orbital angular momenta
    /// $`L = 0, 1, \ldots, L_\text{max}`$.
    ///
    /// A candidate wave is kept when:
    ///
    /// 1. $`L`$ and $`S`$ can couple to the parent $`J`$.
    /// 2. The enabled [`RuleSet`] checks do not reject it.
    ///
    /// Returns an empty vector if the parent spin or either daughter spin is
    /// unknown.
    ///
    /// The returned [`AllowedPartialWave`] includes the underlying
    /// [`PartialWave`] together with channel-dependent inferred quantum numbers,
    /// such as final-state parity and, when meaningful, $`C`$-parity.
    pub fn allowed_partial_waves(
        &self,
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
    ) -> Vec<AllowedPartialWave> {
        let Some(parent_j) = parent.spin else {
            return vec![];
        };
        let Some(ja) = daughters.0.spin else {
            return vec![];
        };
        let Some(jb) = daughters.1.spin else {
            return vec![];
        };
        let mut out = Vec::new();
        for s in Self::coupled_spins(ja, jb) {
            for l_raw in 0..=self.max_l.value() {
                let l = OrbitalAngularMomentum::integer(l_raw);
                let wave = PartialWave::new(parent_j, l, s);
                if let Ok(wave) = wave {
                    // TODO: replace with let-chain in 2024 Rust
                    if self.rules.check(parent, daughters, l, s) {
                        out.push(AllowedPartialWave::new(wave, daughters));
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Charge, Isospin, Projection};

    fn j(twice: u32) -> AngularMomentum {
        AngularMomentum::half_integer(twice)
    }

    fn l(value: u32) -> OrbitalAngularMomentum {
        OrbitalAngularMomentum::integer(value)
    }

    fn q(thirds: i32) -> Charge {
        Charge::third_integer(thirds)
    }

    fn labels(waves: &[AllowedPartialWave]) -> Vec<String> {
        waves.iter().map(|w| w.wave.label.clone()).collect()
    }

    #[test]
    fn coupled_spins_include_all_allowed_values() {
        assert_eq!(SelectionRules::coupled_spins(j(1), j(1)), vec![j(0), j(2)]);
        assert_eq!(SelectionRules::coupled_spins(j(1), j(2)), vec![j(1), j(3)]);
        assert_eq!(
            SelectionRules::coupled_spins(j(2), j(2)),
            vec![j(0), j(2), j(4)]
        );
    }

    #[test]
    fn parity_check_uses_both_daughter_parities() {
        let parent = ParticleProperties::jp(j(0), Parity::Positive);
        let a = ParticleProperties::jp(j(0), Parity::Positive);
        let b = ParticleProperties::jp(j(0), Parity::Negative);
        assert_eq!(RuleSet::check_parity(&parent, (&a, &b), l(0)), Some(false));
        assert_eq!(RuleSet::check_parity(&parent, (&a, &b), l(1)), Some(true));
    }

    #[test]
    fn parity_check_returns_none_when_required_values_are_unknown() {
        let parent = ParticleProperties::unknown();
        let a = ParticleProperties::jp(j(0), Parity::Positive);
        let b = ParticleProperties::jp(j(0), Parity::Negative);
        assert_eq!(RuleSet::check_parity(&parent, (&a, &b), l(0)), None);
    }

    #[test]
    fn additive_checks_reject_any_known_violation() {
        let rules = RuleSet {
            charge: true,
            strangeness: true,
            baryon_number: true,
            ..RuleSet::default()
        };
        let parent = ParticleProperties::unknown()
            .with_charge(q(0))
            .with_strangeness(0)
            .with_baryon_number(0);
        let a = ParticleProperties::unknown()
            .with_charge(q(3))
            .with_strangeness(0)
            .with_baryon_number(0);
        let b = ParticleProperties::unknown()
            .with_charge(q(0))
            .with_strangeness(0)
            .with_baryon_number(0);
        assert_eq!(rules.check_additives(&parent, (&a, &b)), Some(false));
    }

    #[test]
    fn additive_checks_return_none_for_unknowns_only_when_no_violation_is_known() {
        let rules = RuleSet {
            charge: true,
            strangeness: true,
            ..RuleSet::default()
        };
        let parent = ParticleProperties::unknown().with_charge(q(0));
        let a = ParticleProperties::unknown().with_charge(q(3));
        let b = ParticleProperties::unknown().with_charge(q(-3));
        assert_eq!(rules.check_additives(&parent, (&a, &b)), None);
    }

    #[test]
    fn additive_checks_return_some_true_when_all_enabled_checks_pass() {
        let rules = RuleSet {
            charge: true,
            strangeness: true,
            baryon_number: true,
            ..RuleSet::default()
        };
        let parent = ParticleProperties::unknown()
            .with_charge(q(0))
            .with_strangeness(0)
            .with_baryon_number(0);
        let a = ParticleProperties::unknown()
            .with_charge(q(3))
            .with_strangeness(1)
            .with_baryon_number(0);
        let b = ParticleProperties::unknown()
            .with_charge(q(-3))
            .with_strangeness(-1)
            .with_baryon_number(0);
        assert_eq!(rules.check_additives(&parent, (&a, &b)), Some(true));
    }

    #[test]

    fn total_lepton_number_can_pass_when_individual_flavors_change() {
        let rules = RuleSet {
            lepton_number: true,
            electron_lepton_number: false,
            muon_lepton_number: false,
            tau_lepton_number: false,
            ..RuleSet::default()
        };
        let parent = ParticleProperties::unknown()
            .with_electron_lepton_number(1)
            .with_muon_lepton_number(0)
            .with_tau_lepton_number(0);
        let a = ParticleProperties::unknown()
            .with_electron_lepton_number(0)
            .with_muon_lepton_number(1)
            .with_tau_lepton_number(0);
        let b = ParticleProperties::unknown()
            .with_electron_lepton_number(0)
            .with_muon_lepton_number(0)
            .with_tau_lepton_number(0);
        assert_eq!(rules.check_additives(&parent, (&a, &b)), Some(true));
    }

    #[test]
    fn individual_lepton_number_can_reject_flavor_change() {
        let rules = RuleSet {
            lepton_number: true,
            electron_lepton_number: true,
            muon_lepton_number: true,
            tau_lepton_number: true,
            ..RuleSet::default()
        };
        let parent = ParticleProperties::unknown()
            .with_electron_lepton_number(1)
            .with_muon_lepton_number(0)
            .with_tau_lepton_number(0);
        let a = ParticleProperties::unknown()
            .with_electron_lepton_number(0)
            .with_muon_lepton_number(1)
            .with_tau_lepton_number(0);
        let b = ParticleProperties::unknown()
            .with_electron_lepton_number(0)
            .with_muon_lepton_number(0)
            .with_tau_lepton_number(0);
        assert_eq!(rules.check_additives(&parent, (&a, &b)), Some(false));
    }

    #[test]
    fn isospin_coupling_accepts_allowed_parent_isospin() {
        let parent = ParticleProperties::unknown().with_isospin(Isospin::new(j(2), None).unwrap()); // I = 1
        let a = ParticleProperties::unknown().with_isospin(Isospin::new(j(1), None).unwrap()); // I = 1/2
        let b = ParticleProperties::unknown().with_isospin(Isospin::new(j(1), None).unwrap()); // I = 1/2
        assert_eq!(RuleSet::check_isospin(&parent, (&a, &b)), Some(true));
    }

    #[test]
    fn isospin_coupling_rejects_disallowed_parent_isospin() {
        let parent = ParticleProperties::unknown().with_isospin(Isospin::new(j(4), None).unwrap()); // I = 2
        let a = ParticleProperties::unknown().with_isospin(Isospin::new(j(1), None).unwrap()); // I = 1/2
        let b = ParticleProperties::unknown().with_isospin(Isospin::new(j(1), None).unwrap()); // I = 1/2
        assert_eq!(RuleSet::check_isospin(&parent, (&a, &b)), Some(false));
    }

    #[test]
    fn isospin_projection_checks_i3_conservation() {
        let parent = ParticleProperties::unknown()
            .with_isospin(Isospin::new(j(2), Some(Projection::integer(0))).unwrap());
        let a = ParticleProperties::unknown()
            .with_isospin(Isospin::new(j(1), Some(Projection::half_integer(1))).unwrap());
        let b = ParticleProperties::unknown()
            .with_isospin(Isospin::new(j(1), Some(Projection::half_integer(-1))).unwrap());
        assert_eq!(
            RuleSet::check_isospin_projection(&parent, (&a, &b)),
            Some(true)
        );
    }

    #[test]
    fn c_parity_uses_l_plus_s_for_particle_antiparticle_pair() {
        let parent = ParticleProperties::jpc(j(2), Parity::Negative, Parity::Negative);
        let a = ParticleProperties::jp(j(0), Parity::Negative)
            .with_species("pi+")
            .with_antiparticle_species("pi-");
        let b = ParticleProperties::jp(j(0), Parity::Negative)
            .with_species("pi-")
            .with_antiparticle_species("pi+");
        // L = 1, S = 0 -> C = (-1)^(1 + 0) = -
        assert_eq!(
            RuleSet::check_c_parity(&parent, (&a, &b), l(1), j(0)),
            Some(true)
        );
        // L = 0, S = 0 -> C = +
        assert_eq!(
            RuleSet::check_c_parity(&parent, (&a, &b), l(0), j(0)),
            Some(false)
        );
    }

    #[test]
    fn g_parity_checks_product_of_daughter_g_parities() {
        let parent = ParticleProperties::unknown().with_g_parity(Parity::Positive);
        let a = ParticleProperties::unknown().with_g_parity(Parity::Negative);
        let b = ParticleProperties::unknown().with_g_parity(Parity::Negative);
        assert_eq!(RuleSet::check_g_parity(&parent, (&a, &b)), Some(true));
    }

    #[test]
    fn identical_spin_zero_bosons_require_even_l() {
        let a = ParticleProperties::jp(j(0), Parity::Negative)
            .with_species("pi0")
            .with_statistics(Statistics::Boson)
            .unwrap();
        let b = ParticleProperties::jp(j(0), Parity::Negative)
            .with_species("pi0")
            .with_statistics(Statistics::Boson)
            .unwrap();
        assert_eq!(
            RuleSet::check_identical_particle_symmetry((&a, &b), l(0)),
            Some(true)
        );
        assert_eq!(
            RuleSet::check_identical_particle_symmetry((&a, &b), l(1)),
            Some(false)
        );
    }

    #[test]
    fn selection_rules_find_delta_to_n_pi_p_wave() {
        let parent = ParticleProperties::jp(j(3), Parity::Positive)
            .with_charge(q(3))
            .with_baryon_number(1);
        let nucleon = ParticleProperties::jp(j(1), Parity::Positive)
            .with_charge(q(3))
            .with_baryon_number(1);
        let pion = ParticleProperties::jp(j(0), Parity::Negative)
            .with_charge(q(0))
            .with_baryon_number(0);
        let rules = SelectionRules {
            max_l: l(4),
            rules: RuleSet {
                parity: true,
                charge: true,
                baryon_number: true,
                ..RuleSet::angular()
            },
        };
        let waves = rules.allowed_partial_waves(&parent, (&nucleon, &pion));
        assert_eq!(labels(&waves), vec!["2P3/2"]);
    }

    #[test]
    fn angular_only_selection_rules_include_all_l_s_couplings() {
        let parent = ParticleProperties::jp(j(2), Parity::Positive); // J = 1
        let a = ParticleProperties::jp(j(1), Parity::Positive); // spin 1/2
        let b = ParticleProperties::jp(j(1), Parity::Negative); // spin 1/2
        let rules = SelectionRules {
            max_l: l(2),
            rules: RuleSet::angular(),
        };
        let waves = rules.allowed_partial_waves(&parent, (&a, &b));
        let got = labels(&waves);
        assert_eq!(got, vec!["1P1", "3S1", "3P1", "3D1"]);
    }

    #[test]
    fn strong_parity_filter_removes_wrong_l_values() {
        let parent = ParticleProperties::jp(j(2), Parity::Positive); // J^P = 1+
        let a = ParticleProperties::jp(j(1), Parity::Positive);
        let b = ParticleProperties::jp(j(1), Parity::Negative);
        let rules = SelectionRules {
            max_l: l(2),
            rules: RuleSet {
                parity: true,
                ..RuleSet::angular()
            },
        };
        let waves = rules.allowed_partial_waves(&parent, (&a, &b));
        let got = labels(&waves);
        // P_parent = +, P_a P_b = -, so L must be odd.
        assert_eq!(got, vec!["1P1", "3P1"]);
    }

    #[test]
    fn allowed_partial_waves_returns_empty_when_spin_information_is_missing() {
        let parent = ParticleProperties::unknown();
        let a = ParticleProperties::jp(j(0), Parity::Negative);
        let b = ParticleProperties::jp(j(0), Parity::Negative);
        let rules = SelectionRules::default();
        assert!(rules.allowed_partial_waves(&parent, (&a, &b)).is_empty());
    }
}
