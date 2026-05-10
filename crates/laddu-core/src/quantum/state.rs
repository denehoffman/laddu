use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::{
    quantum::types::Statistics, AngularMomentum, Charge, LadduError, LadduResult,
    OrbitalAngularMomentum, Parity, Projection,
};

/// A validated spin state with spin and projection stored as doubled quantum numbers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct SpinState {
    spin: AngularMomentum,
    projection: Projection,
}

impl SpinState {
    /// Construct a spin state after validating projection bounds and parity.
    pub fn new(spin: AngularMomentum, projection: Projection) -> LadduResult<Self> {
        validate_projection(spin, projection)?;
        Ok(Self { spin, projection })
    }

    /// Return the spin quantum number.
    pub const fn spin(self) -> AngularMomentum {
        self.spin
    }

    /// Return the spin projection quantum number.
    pub const fn projection(self) -> Projection {
        self.projection
    }

    /// Enumerate all allowed projections for `spin`.
    pub fn allowed_projections(spin: AngularMomentum) -> Vec<Self> {
        let spin_value = spin.value() as i32;
        (-spin_value..=spin_value)
            .step_by(2)
            .map(|projection| Self {
                spin,
                projection: Projection::half_integer(projection),
            })
            .collect()
    }
}

/// An isospin state with optional projection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Isospin {
    isospin: AngularMomentum,
    projection: Option<Projection>,
}

impl Isospin {
    /// Construct a new isospin state from the given total isospin and optional projection.
    pub fn new(isospin: AngularMomentum, projection: Option<Projection>) -> LadduResult<Self> {
        if let Some(projection) = projection {
            validate_projection(isospin, projection)?;
        }
        Ok(Self {
            isospin,
            projection,
        })
    }
    /// The total isospin of the state.
    pub fn isospin(self) -> AngularMomentum {
        self.isospin
    }
    /// The isospin projection of the state.
    pub fn projection(self) -> Option<Projection> {
        self.projection
    }
}

/// The set of properties which define the quantum state of a particle.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ParticleProperties {
    /// The name of the particle, if known.
    pub name: Option<String>,
    /// The species of the particle, if known (used to compare to [`ParticleProperties::antiparticle_species`]).
    pub species: Option<String>,
    /// The species of the particle's antiparticle, if known (used to compare to [`ParticleProperties::species`]).
    pub antiparticle_species: Option<String>,
    /// Whether the particle is its own antiparticle.
    pub self_conjugate: Option<bool>,
    /// The spin of the particle, if known.
    pub spin: Option<AngularMomentum>,
    /// The intrinsic parity of the particle, if known.
    pub parity: Option<Parity>,
    /// The intrinsic C-parity of the particle, if known or applicable.
    pub c_parity: Option<Parity>,
    /// The intrinsic G-parity of the particle, if known or applicable.
    pub g_parity: Option<Parity>,
    /// The electric charge of the particle, if known.
    pub charge: Option<Charge>,
    /// The isospin of the particle, if known.
    pub isospin: Option<Isospin>,
    /// The total strangeness of the particle, if known.
    pub strangeness: Option<i32>,
    /// The total charm of the particle, if known.
    pub charm: Option<i32>,
    /// The total bottomness of the particle, if known.
    pub bottomness: Option<i32>,
    /// The total topness of the particle, if known.
    pub topness: Option<i32>,
    /// The total baryon number of the particle, if known.
    pub baryon_number: Option<i32>,
    /// The electron lepton number of the particle, if known.
    pub electron_lepton_number: Option<i32>,
    /// The muon lepton number of the particle, if known.
    pub muon_lepton_number: Option<i32>,
    /// The tau lepton number of the particle, if known.
    pub tau_lepton_number: Option<i32>,
    /// The particle's statistical nature, if known.
    pub statistics: Option<Statistics>,
}

impl ParticleProperties {
    /// Construct a particle with no specified properties.
    pub fn unknown() -> Self {
        Self::default()
    }

    /// Construct a particle with the given spin and parity.
    pub fn jp(j: AngularMomentum, p: Parity) -> Self {
        Self {
            spin: Some(j),
            parity: Some(p),
            statistics: Some(Statistics::from_spin(j)),
            ..Self::default()
        }
    }
    /// Construct a particle with the given spin, parity, and C-parity.
    pub fn jpc(j: AngularMomentum, p: Parity, c: Parity) -> Self {
        Self {
            spin: Some(j),
            parity: Some(p),
            c_parity: Some(c),
            statistics: Some(Statistics::from_spin(j)),
            ..Self::default()
        }
    }
    /// Set the particle's name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    /// Set the particle's species.
    pub fn with_species(mut self, species: impl Into<String>) -> Self {
        self.species = Some(species.into());
        self
    }
    /// Set the particle's antiparticle species.
    pub fn with_antiparticle_species(mut self, antiparticle_species: impl Into<String>) -> Self {
        self.antiparticle_species = Some(antiparticle_species.into());
        self
    }
    /// Set whether the particle is its own antiparticle.
    pub fn with_self_conjugate(mut self, value: bool) -> Self {
        self.self_conjugate = Some(value);
        self
    }
    /// Set the particle's spin.
    pub fn with_spin(mut self, j: AngularMomentum) -> Self {
        self.spin = Some(j);
        self.statistics = Some(Statistics::from_spin(j));
        self
    }
    /// Set the particle's intrinsic parity.
    pub fn with_parity(mut self, p: Parity) -> Self {
        self.parity = Some(p);
        self
    }
    /// Set the particle's intrinsic C-parity.
    pub fn with_c_parity(mut self, c: Parity) -> Self {
        self.c_parity = Some(c);
        self
    }
    /// Set the particle's intrinsic G-parity.
    pub fn with_g_parity(mut self, g: Parity) -> Self {
        self.g_parity = Some(g);
        self
    }
    /// Set the particle's electric charge.
    pub fn with_charge(mut self, q: Charge) -> Self {
        self.charge = Some(q);
        self
    }
    /// Set the particle's isospin state.
    pub fn with_isospin(mut self, isospin: Isospin) -> Self {
        self.isospin = Some(isospin);
        self
    }
    /// Set the particle's total strangeness.
    pub fn with_strangeness(mut self, s: i32) -> Self {
        self.strangeness = Some(s);
        self
    }
    /// Set the particle's total charm.
    pub fn with_charm(mut self, c: i32) -> Self {
        self.charm = Some(c);
        self
    }
    /// Set the particle's total bottomness.
    pub fn with_bottomness(mut self, b: i32) -> Self {
        self.bottomness = Some(b);
        self
    }
    /// Set the particle's total topness.
    pub fn with_topness(mut self, t: i32) -> Self {
        self.topness = Some(t);
        self
    }
    /// Set the particle's total baryon number.
    pub fn with_baryon_number(mut self, b: i32) -> Self {
        self.baryon_number = Some(b);
        self
    }
    /// Set the particle's electron lepton number.
    pub fn with_electron_lepton_number(mut self, e: i32) -> Self {
        self.electron_lepton_number = Some(e);
        self
    }
    /// Set the particle's muon lepton number.
    pub fn with_muon_lepton_number(mut self, m: i32) -> Self {
        self.muon_lepton_number = Some(m);
        self
    }
    /// Set the particle's tau lepton number.
    pub fn with_tau_lepton_number(mut self, t: i32) -> Self {
        self.tau_lepton_number = Some(t);
        self
    }
    /// Set the particle's statistical nature.
    ///
    /// Returns an error if the spin and statistics do not match.
    pub fn with_statistics(mut self, s: Statistics) -> LadduResult<Self> {
        if let Some(spin) = self.spin {
            if Statistics::from_spin(spin) != s {
                return Err(LadduError::Custom(
                    "spin and statistics must be consistent".to_string(),
                ));
            }
        }
        self.statistics = Some(s);
        Ok(self)
    }

    /// Returns true if `self` is the antiparticle of `other`.
    pub fn is_antiparticle_of(&self, other: &ParticleProperties) -> bool {
        let a_species = self.species.as_ref();
        let b_species = other.species.as_ref();

        let a_anti = self.antiparticle_species.as_ref();
        let b_anti = other.antiparticle_species.as_ref();

        match (a_species, b_species, a_anti, b_anti) {
            (Some(a), Some(b), Some(a_bar), Some(b_bar)) => a_bar == b && b_bar == a,
            (Some(_), Some(b), Some(a_bar), None) => a_bar == b,
            (Some(a), Some(_), None, Some(b_bar)) => b_bar == a,
            _ => false,
        }
    }
}

/// A partial wave defined by a total angular momentum, `J`, an orbital angular momentum, `L`, and
/// and intrinsic spin, `S`.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PartialWave {
    /// The total angular momentum of the wave
    pub j: AngularMomentum,
    /// The orbital angular momentum of the wave
    pub l: OrbitalAngularMomentum,
    /// The spin of the wave
    pub s: AngularMomentum,
    /// The spectroscopic label of the wave
    pub label: String,
}
impl PartialWave {
    /// Construct a new partial wave from the given angular momentum quantum numbers.
    pub fn new(
        j: AngularMomentum,
        l: OrbitalAngularMomentum,
        s: AngularMomentum,
    ) -> LadduResult<Self> {
        PartialWave::validate_coupling(j, l, s)?;
        let multiplicity = s.value() + 1;
        Ok(Self {
            j,
            l,
            s,
            label: format!("{}{}{}", multiplicity, l, j),
        })
    }
    /// Set the spectroscopic label of the wave.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }
    /// Validate the set of angular momentum quantum numbers which define a partial wave.
    pub fn validate_coupling(
        j: AngularMomentum,
        l: OrbitalAngularMomentum,
        s: AngularMomentum,
    ) -> LadduResult<()> {
        let l_twice = 2 * l.value();
        let s_twice = s.value();
        let j_twice = j.value();
        let min = l_twice.abs_diff(s_twice);
        let max = l_twice + s_twice;
        if j_twice >= min && j_twice <= max && (j_twice - min).is_multiple_of(2) {
            Ok(())
        } else {
            Err(LadduError::Custom(
                "j, l, and s must be compatible".to_string(),
            ))
        }
    }
}

impl Display for PartialWave {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

/// A partial wave together with allowed parity and C-parity, if applicable.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct AllowedPartialWave {
    /// The angular quantum numbers of the wave
    pub wave: PartialWave,
    /// The allowed parity, if applicable
    pub parity: Option<Parity>,
    /// The allowed C-parity, if applicable
    pub c_parity: Option<Parity>,
}

impl AllowedPartialWave {
    /// Take an existing [`PartialWave`] and infer parity and C-parity from its decay products.
    pub fn new(wave: PartialWave, daughters: (&ParticleProperties, &ParticleProperties)) -> Self {
        Self {
            parity: Self::infer_parity(daughters, wave.l),
            c_parity: Self::infer_c_parity(daughters, wave.l, wave.s),
            wave,
        }
    }

    /// Infer the parity of a state given the parity of its decay products and its orbital angular momentum.
    pub fn infer_parity(
        daughters: (&ParticleProperties, &ParticleProperties),
        l: OrbitalAngularMomentum,
    ) -> Option<Parity> {
        let p_a = daughters.0.parity?;
        let p_b = daughters.1.parity?;

        let value = p_a.value() * p_b.value() * if l.value() & 1 == 0 { 1 } else { -1 };

        Some(if value == 1 {
            Parity::Positive
        } else {
            Parity::Negative
        })
    }

    /// Infer the C-parity of a state given the species of its decay products, its orbital angular momentum, and its intrinsic spin.
    pub fn infer_c_parity(
        daughters: (&ParticleProperties, &ParticleProperties),
        l: OrbitalAngularMomentum,
        s: AngularMomentum,
    ) -> Option<Parity> {
        if !daughters.0.is_antiparticle_of(daughters.1) {
            return None;
        }

        let exp_twice = 2 * l.value() + s.value();

        if !exp_twice.is_multiple_of(2) {
            return None;
        }

        Some(if (exp_twice / 2).is_multiple_of(2) {
            Parity::Positive
        } else {
            Parity::Negative
        })
    }
}

fn validate_projection(spin: AngularMomentum, projection: Projection) -> LadduResult<()> {
    if projection.value().unsigned_abs() > spin.value() {
        return Err(LadduError::Custom(
            "spin projection must satisfy -J <= m <= J".to_string(),
        ));
    }
    if !spin.has_same_parity_as(projection) {
        return Err(LadduError::Custom(
            "spin projection must have the same integer or half-integer parity as spin".to_string(),
        ));
    }
    Ok(())
}
