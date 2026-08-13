use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    LadduPhysicsError, LadduPhysicsResult,
    quantum::{J, L, M, Parity, Statistics},
};

/// An external identifier associated with a physical particle species.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ExternalId {
    /// A numeric identifier, such as a PDG code.
    Code {
        /// Identifier value.
        value: i64,
    },
    /// A textual identifier.
    Label {
        /// Identifier value.
        value: String,
    },
}

impl From<&str> for ExternalId {
    fn from(value: &str) -> Self {
        Self::label(value)
    }
}
impl From<String> for ExternalId {
    fn from(value: String) -> Self {
        Self::label(value)
    }
}
impl From<&String> for ExternalId {
    fn from(value: &String) -> Self {
        Self::label(value)
    }
}
impl From<i64> for ExternalId {
    fn from(value: i64) -> Self {
        Self::code(value)
    }
}

impl ExternalId {
    /// Construct a numeric identifier.
    pub fn code(value: i64) -> Self {
        Self::Code { value }
    }

    /// Construct a textual identifier in an arbitrary namespace.
    pub fn label(value: impl Into<String>) -> Self {
        Self::Label {
            value: value.into(),
        }
    }

    /// Return the numeric value, if this is a numeric identifier.
    pub fn code_value(&self) -> Option<i64> {
        match self {
            Self::Code { value, .. } => Some(*value),
            Self::Label { .. } => None,
        }
    }

    /// Return the textual value, if this is a textual identifier.
    pub fn label_value(&self) -> Option<&str> {
        match self {
            Self::Label { value, .. } => Some(value),
            Self::Code { .. } => None,
        }
    }
}

/// A validated spin state with spin and projection stored as doubled quantum numbers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct SpinState {
    spin: J,
    projection: M,
}

impl SpinState {
    /// Construct a spin state after validating projection bounds and parity.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `projection` is outside the spin
    /// range or has incompatible integer/half-integer parity.
    pub fn new(spin: J, projection: M) -> LadduPhysicsResult<Self> {
        validate_projection(spin, projection)?;
        Ok(Self { spin, projection })
    }

    /// Return the spin quantum number.
    pub const fn spin(self) -> J {
        self.spin
    }

    /// Return the spin projection quantum number.
    pub const fn projection(self) -> M {
        self.projection
    }
}

/// An isospin state with optional projection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Isospin {
    /// The total isospin of the state.
    pub isospin: J,
    /// The isospin projection of the state.
    pub projection: Option<M>,
}

impl Isospin {
    /// Construct a new isospin state from the given total isospin and optional projection.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the projection is outside the
    /// isospin range or has incompatible integer/half-integer parity.
    pub fn new(isospin: J, projection: Option<M>) -> LadduPhysicsResult<Self> {
        if let Some(projection) = projection {
            validate_projection(isospin, projection)?;
        }
        Ok(Self {
            isospin,
            projection,
        })
    }

    /// The total isospin of the state.
    pub fn isospin(self) -> J {
        self.isospin
    }
    /// The isospin projection of the state.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the
    /// projection is unknown.
    pub fn projection(self) -> LadduPhysicsResult<M> {
        self.projection
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "isospin.projection",
            })
    }
}

impl From<J> for Isospin {
    fn from(value: J) -> Self {
        Self {
            isospin: value,
            projection: None,
        }
    }
}

/// The set of properties which define the quantum state of a particle.
///
/// Direct assignments to public fields are not validated. Fallible `with_*`
/// methods validate relationships between particle properties; infallible
/// setters update only the named property.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    pub spin: Option<J>,
    /// The intrinsic parity of the particle, if known.
    pub parity: Option<Parity>,
    /// The intrinsic C-parity of the particle, if known or applicable.
    pub c_parity: Option<Parity>,
    /// The intrinsic G-parity of the particle, if known or applicable.
    pub g_parity: Option<Parity>,
    /// The electric charge of the particle, if known.
    pub charge: Option<i32>,
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
    /// The nominal particle mass, if known.
    pub mass: Option<f64>,
    /// External identifiers for this particle.
    pub ids: IndexMap<String, ExternalId>,
}

/// A set of particle properties to apply together.
///
/// Fields set to `None` are left unchanged.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct ParticlePropertiesPatch {
    pub name: Option<String>,
    pub species: Option<String>,
    pub antiparticle_species: Option<String>,
    pub self_conjugate: Option<bool>,
    pub spin: Option<J>,
    pub parity: Option<Parity>,
    pub c_parity: Option<Parity>,
    pub g_parity: Option<Parity>,
    pub charge: Option<i32>,
    pub isospin: Option<Isospin>,
    pub strangeness: Option<i32>,
    pub charm: Option<i32>,
    pub bottomness: Option<i32>,
    pub topness: Option<i32>,
    pub baryon_number: Option<i32>,
    pub electron_lepton_number: Option<i32>,
    pub muon_lepton_number: Option<i32>,
    pub tau_lepton_number: Option<i32>,
    pub statistics: Option<Statistics>,
    pub mass: Option<f64>,
    pub ids: Option<IndexMap<String, ExternalId>>,
}

#[derive(Clone, Copy)]
enum AdditiveQuantumNumber {
    Charge,
    Strangeness,
    Charm,
    Bottomness,
    Topness,
    BaryonNumber,
    ElectronLeptonNumber,
    MuonLeptonNumber,
    TauLeptonNumber,
}

impl AdditiveQuantumNumber {
    fn name(self) -> &'static str {
        match self {
            Self::Charge => "charge",
            Self::Strangeness => "strangeness",
            Self::Charm => "charm",
            Self::Bottomness => "bottomness",
            Self::Topness => "topness",
            Self::BaryonNumber => "baryon_number",
            Self::ElectronLeptonNumber => "electron_lepton_number",
            Self::MuonLeptonNumber => "muon_lepton_number",
            Self::TauLeptonNumber => "tau_lepton_number",
        }
    }

    fn field(self, particle: &mut ParticleProperties) -> &mut Option<i32> {
        match self {
            Self::Charge => &mut particle.charge,
            Self::Strangeness => &mut particle.strangeness,
            Self::Charm => &mut particle.charm,
            Self::Bottomness => &mut particle.bottomness,
            Self::Topness => &mut particle.topness,
            Self::BaryonNumber => &mut particle.baryon_number,
            Self::ElectronLeptonNumber => &mut particle.electron_lepton_number,
            Self::MuonLeptonNumber => &mut particle.muon_lepton_number,
            Self::TauLeptonNumber => &mut particle.tau_lepton_number,
        }
    }
}

enum ParticleUpdate {
    Species(String),
    AntiparticleSpecies(String),
    SpeciesNames(String, String),
    SelfConjugate(bool),
    SelfConjugateSpecies(String),
    CParity(Parity),
    Additive(AdditiveQuantumNumber, i32),
    Statistics(Statistics),
}

impl ParticleProperties {
    /// Get the particle's name
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the name is
    /// unknown.
    pub fn name(&self) -> LadduPhysicsResult<String> {
        self.name
            .clone()
            .ok_or(LadduPhysicsError::MissingParticleProperty { property: "name" })
            .clone()
    }
    /// Get the particle's species
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the species
    /// is unknown.
    pub fn species(&self) -> LadduPhysicsResult<String> {
        self.species
            .clone()
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "species",
            })
            .clone()
    }
    /// Get the particle's antiparticle species
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the
    /// antiparticle species is unknown.
    pub fn antiparticle_species(&self) -> LadduPhysicsResult<String> {
        self.antiparticle_species
            .clone()
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "antiparticle_species",
            })
            .clone()
    }
    /// Get the particle's self-conjugate status
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the
    /// self-conjugate status is unknown.
    pub fn self_conjugate(&self) -> LadduPhysicsResult<bool> {
        self.self_conjugate
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "self_conjugate",
            })
            .clone()
    }
    /// Get the particle's spin
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the spin is
    /// unknown.
    pub fn spin(&self) -> LadduPhysicsResult<J> {
        self.spin
            .ok_or(LadduPhysicsError::MissingParticleProperty { property: "spin" })
            .clone()
    }
    /// Get the particle's intrinsic parity
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when parity is
    /// unknown.
    pub fn parity(&self) -> LadduPhysicsResult<Parity> {
        self.parity
            .ok_or(LadduPhysicsError::MissingParticleProperty { property: "parity" })
            .clone()
    }
    /// Get the particle's intrinsic C-parity
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when C-parity is
    /// unknown.
    pub fn c_parity(&self) -> LadduPhysicsResult<Parity> {
        self.c_parity
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "c_parity",
            })
            .clone()
    }
    /// Get the particle's intrinsic G-parity
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when G-parity is
    /// unknown.
    pub fn g_parity(&self) -> LadduPhysicsResult<Parity> {
        self.g_parity
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "g_parity",
            })
            .clone()
    }
    /// Get the particle's electric charge
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when charge is
    /// unknown.
    pub fn charge(&self) -> LadduPhysicsResult<i32> {
        self.charge
            .ok_or(LadduPhysicsError::MissingParticleProperty { property: "charge" })
            .clone()
    }
    /// Get the particle's isospin
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when isospin is
    /// unknown.
    pub fn isospin(&self) -> LadduPhysicsResult<Isospin> {
        self.isospin
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "isospin",
            })
            .clone()
    }
    /// Get the particle's strangeness
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when strangeness
    /// is unknown.
    pub fn strangeness(&self) -> LadduPhysicsResult<i32> {
        self.strangeness
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "strangeness",
            })
            .clone()
    }
    /// Get the particle's charm
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when charm is
    /// unknown.
    pub fn charm(&self) -> LadduPhysicsResult<i32> {
        self.charm
            .ok_or(LadduPhysicsError::MissingParticleProperty { property: "charm" })
            .clone()
    }
    /// Get the particle's bottomness
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when bottomness
    /// is unknown.
    pub fn bottomness(&self) -> LadduPhysicsResult<i32> {
        self.bottomness
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "bottomness",
            })
            .clone()
    }
    /// Get the particle's topness
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when topness is
    /// unknown.
    pub fn topness(&self) -> LadduPhysicsResult<i32> {
        self.topness
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "topness",
            })
            .clone()
    }
    /// Get the particle's baryon number
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the baryon
    /// number is unknown.
    pub fn baryon_number(&self) -> LadduPhysicsResult<i32> {
        self.baryon_number
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "baryon_number",
            })
            .clone()
    }
    /// Get the particle's electron lepton number
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the electron
    /// lepton number is unknown.
    pub fn electron_lepton_number(&self) -> LadduPhysicsResult<i32> {
        self.electron_lepton_number
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "electron_lepton_number",
            })
            .clone()
    }
    /// Get the particle's muon lepton number
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the muon
    /// lepton number is unknown.
    pub fn muon_lepton_number(&self) -> LadduPhysicsResult<i32> {
        self.muon_lepton_number
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "muon_lepton_number",
            })
            .clone()
    }
    /// Get the particle's tau lepton number
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the tau
    /// lepton number is unknown.
    pub fn tau_lepton_number(&self) -> LadduPhysicsResult<i32> {
        self.tau_lepton_number
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "tau_lepton_number",
            })
            .clone()
    }
    /// Get the particle's statistics
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the
    /// statistics are unknown.
    pub fn statistics(&self) -> LadduPhysicsResult<Statistics> {
        self.statistics
            .ok_or(LadduPhysicsError::MissingParticleProperty {
                property: "statistics",
            })
            .clone()
    }
    /// Get the particle's mass.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError::MissingParticleProperty`] when the mass is
    /// unknown.
    pub fn mass(&self) -> LadduPhysicsResult<f64> {
        self.mass
            .ok_or(LadduPhysicsError::MissingParticleProperty { property: "mass" })
    }

    /// Construct a particle with no specified properties.
    pub fn unknown() -> Self {
        Self::default()
    }

    /// Apply the provided properties and validate relationships between them.
    #[doc(hidden)]
    pub fn apply_patch(mut self, patch: ParticlePropertiesPatch) -> LadduPhysicsResult<Self> {
        let ParticlePropertiesPatch {
            name,
            species,
            antiparticle_species,
            self_conjugate,
            spin,
            parity,
            c_parity,
            g_parity,
            charge,
            isospin,
            strangeness,
            charm,
            bottomness,
            topness,
            baryon_number,
            electron_lepton_number,
            muon_lepton_number,
            tau_lepton_number,
            statistics,
            mass,
            ids,
        } = patch;

        if let Some(name) = name {
            self = self.with_name(name);
        }
        self = match (species, antiparticle_species) {
            (Some(species), Some(antiparticle)) => {
                self.with_species_names(species, antiparticle)?
            }
            (Some(species), None) => self.with_species(species)?,
            (None, Some(antiparticle)) => self.with_antiparticle_species(antiparticle)?,
            (None, None) => self,
        };
        if let Some(value) = self_conjugate {
            self = self.with_self_conjugate(value)?;
        }
        if let Some(value) = spin {
            self = self.with_spin(value);
        }
        if let Some(value) = parity {
            self = self.with_parity(value);
        }
        if let Some(value) = c_parity {
            self = self.with_c_parity(value)?;
        }
        if let Some(value) = g_parity {
            self = self.with_g_parity(value);
        }
        if let Some(value) = charge {
            self = self.apply_update(ParticleUpdate::Additive(
                AdditiveQuantumNumber::Charge,
                value,
            ))?;
        }
        if let Some(value) = isospin {
            self = self.with_isospin(value);
        }
        if let Some(value) = strangeness {
            self = self.with_strangeness(value)?;
        }
        if let Some(value) = charm {
            self = self.with_charm(value)?;
        }
        if let Some(value) = bottomness {
            self = self.with_bottomness(value)?;
        }
        if let Some(value) = topness {
            self = self.with_topness(value)?;
        }
        if let Some(value) = baryon_number {
            self = self.with_baryon_number(value)?;
        }
        if let Some(value) = electron_lepton_number {
            self = self.with_electron_lepton_number(value)?;
        }
        if let Some(value) = muon_lepton_number {
            self = self.with_muon_lepton_number(value)?;
        }
        if let Some(value) = tau_lepton_number {
            self = self.with_tau_lepton_number(value)?;
        }
        if let Some(value) = statistics {
            self = self.with_statistics(value)?;
        }
        if let Some(value) = mass {
            if !value.is_finite() || value < 0.0 {
                return Err(LadduPhysicsError::invalid_value(
                    "particle mass",
                    "finite and non-negative",
                    value,
                ));
            }
            self = self.with_mass(value);
        }
        if let Some(value) = ids {
            self.ids = value;
        }

        Ok(self)
    }

    /// Construct a particle with the given spin and parity.
    /// Construct a particle with the given spin and intrinsic parity.
    pub fn jp(j: J, p: Parity) -> Self {
        Self {
            spin: Some(j),
            parity: Some(p),
            statistics: Some(Statistics::from_spin(j)),
            ..Self::default()
        }
    }
    /// Construct a particle with the given spin, parity, and C-parity.
    pub fn jpc(j: J, p: Parity, c: Parity) -> Self {
        Self {
            spin: Some(j),
            parity: Some(p),
            c_parity: Some(c),
            statistics: Some(Statistics::from_spin(j)),
            ..Self::default()
        }
    }

    /// A Boson-like state with spin `j` and zero baryon or lepton number
    pub fn boson(j: L) -> Self {
        let mut particle = Self::unknown().with_spin(j.into());
        particle.baryon_number = Some(0);
        particle.electron_lepton_number = Some(0);
        particle.muon_lepton_number = Some(0);
        particle.tau_lepton_number = Some(0);
        particle
    }

    /// Construct a lepton-like state with the supplied family lepton numbers.
    pub fn lepton(e: i32, m: i32, t: i32) -> Self {
        let mut particle = Self::unknown().with_zero_flavor();
        particle.baryon_number = Some(0);
        particle.electron_lepton_number = Some(e);
        particle.muon_lepton_number = Some(m);
        particle.tau_lepton_number = Some(t);
        particle
    }

    /// A hadron-like state with zero lepton number.
    /// Does not assume baryon number, charge, or flavor.
    pub fn hadron() -> Self {
        Self::unknown().with_zero_lepton_numbers()
    }

    /// A meson-like hadron with zero baryon and lepton number.
    /// Does not assume charge or flavor.
    pub fn meson() -> Self {
        let mut particle = Self::hadron();
        particle.baryon_number = Some(0);
        particle
    }

    /// A baryon-like hadron with baryon number `b` and zero lepton number.
    /// Usually `b = 1`; nuclei/dibaryons can use `b > 1`; antibaryons use negative values.
    pub fn baryon(b: i32) -> Self {
        let mut particle = Self::hadron();
        particle.baryon_number = Some(b);
        particle
    }

    /// Set the particle's name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    /// Set the particle's species.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the species conflicts with existing
    /// self-conjugacy or antiparticle metadata.
    pub fn with_species(self, species: impl Into<String>) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::Species(species.into()))
    }
    /// Set the particle's antiparticle species.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the antiparticle species conflicts
    /// with existing self-conjugacy or species metadata.
    pub fn with_antiparticle_species(
        self,
        antiparticle_species: impl Into<String>,
    ) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::AntiparticleSpecies(
            antiparticle_species.into(),
        ))
    }

    /// Set both particle and antiparticle species names.
    ///
    /// Equal names mark the particle as self-conjugate.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the resulting species and quantum
    /// number metadata violate particle invariants.
    pub fn with_species_names(
        self,
        species: impl Into<String>,
        antiparticle_species: impl Into<String>,
    ) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::SpeciesNames(
            species.into(),
            antiparticle_species.into(),
        ))
    }

    /// Set whether the particle is its own antiparticle.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `value` conflicts with species names,
    /// C-parity, or nonzero additive quantum numbers.
    pub fn with_self_conjugate(self, value: bool) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::SelfConjugate(value))
    }

    /// Set one species name and mark the particle as self-conjugate.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when existing particle metadata is
    /// inconsistent with self-conjugacy.
    pub fn with_self_conjugate_species(
        self,
        species: impl Into<String>,
    ) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::SelfConjugateSpecies(species.into()))
    }

    /// Set the particle's spin.
    pub fn with_spin(mut self, j: J) -> Self {
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
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when C-parity conflicts with the
    /// particle's self-conjugacy or additive quantum numbers.
    pub fn with_c_parity(self, c: Parity) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::CParity(c))
    }
    /// Set the particle's intrinsic G-parity.
    pub fn with_g_parity(mut self, g: Parity) -> Self {
        self.g_parity = Some(g);
        self
    }
    /// Set the particle's electric charge.
    pub fn with_charge(mut self, q: i32) -> Self {
        self.charge = Some(q);
        self
    }
    /// Set the particle's isospin state.
    pub fn with_isospin(mut self, isospin: Isospin) -> Self {
        self.isospin = Some(isospin);
        self
    }
    /// Set the particle's total strangeness.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when nonzero strangeness conflicts with
    /// self-conjugacy.
    pub fn with_strangeness(self, s: i32) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::Additive(
            AdditiveQuantumNumber::Strangeness,
            s,
        ))
    }
    /// Set the particle's total charm.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when nonzero charm conflicts with
    /// self-conjugacy.
    pub fn with_charm(self, c: i32) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::Additive(AdditiveQuantumNumber::Charm, c))
    }
    /// Set the particle's total bottomness.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when nonzero bottomness conflicts with
    /// self-conjugacy.
    pub fn with_bottomness(self, b: i32) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::Additive(
            AdditiveQuantumNumber::Bottomness,
            b,
        ))
    }
    /// Set the particle's total topness.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when nonzero topness conflicts with
    /// self-conjugacy.
    pub fn with_topness(self, t: i32) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::Additive(AdditiveQuantumNumber::Topness, t))
    }
    /// Set strangeness, charm, bottomness, and topness together.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when a nonzero flavor quantum number
    /// conflicts with self-conjugacy.
    pub fn with_flavor(self, s: i32, c: i32, b: i32, t: i32) -> LadduPhysicsResult<Self> {
        self.with_strangeness(s)?
            .with_charm(c)?
            .with_bottomness(b)?
            .with_topness(t)
    }
    /// Set the particle's total baryon number.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when a nonzero baryon number conflicts
    /// with self-conjugacy.
    pub fn with_baryon_number(self, b: i32) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::Additive(
            AdditiveQuantumNumber::BaryonNumber,
            b,
        ))
    }
    /// Set the particle's electron lepton number.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when a nonzero electron lepton number
    /// conflicts with self-conjugacy.
    pub fn with_electron_lepton_number(self, e: i32) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::Additive(
            AdditiveQuantumNumber::ElectronLeptonNumber,
            e,
        ))
    }
    /// Set the particle's muon lepton number.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when a nonzero muon lepton number
    /// conflicts with self-conjugacy.
    pub fn with_muon_lepton_number(self, m: i32) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::Additive(
            AdditiveQuantumNumber::MuonLeptonNumber,
            m,
        ))
    }
    /// Set the particle's tau lepton number.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when a nonzero tau lepton number
    /// conflicts with self-conjugacy.
    pub fn with_tau_lepton_number(self, t: i32) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::Additive(
            AdditiveQuantumNumber::TauLeptonNumber,
            t,
        ))
    }

    /// Set electron-, muon-, and tau-family lepton numbers together.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when a nonzero lepton number conflicts
    /// with self-conjugacy.
    pub fn with_lepton_numbers(self, e: i32, m: i32, t: i32) -> LadduPhysicsResult<Self> {
        self.with_electron_lepton_number(e)?
            .with_muon_lepton_number(m)?
            .with_tau_lepton_number(t)
    }

    /// Set the particle's statistical nature.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] if the spin and statistics do not match.
    pub fn with_statistics(self, s: Statistics) -> LadduPhysicsResult<Self> {
        self.apply_update(ParticleUpdate::Statistics(s))
    }
    /// Set the particle's mass.
    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = Some(mass);
        self
    }

    /// Set every flavor quantum number to zero.
    pub fn with_zero_flavor(mut self) -> Self {
        self.strangeness = Some(0);
        self.charm = Some(0);
        self.bottomness = Some(0);
        self.topness = Some(0);
        self
    }

    /// Set every family lepton number to zero.
    pub fn with_zero_lepton_numbers(mut self) -> Self {
        self.electron_lepton_number = Some(0);
        self.muon_lepton_number = Some(0);
        self.tau_lepton_number = Some(0);
        self
    }

    /// Set charge, flavor, baryon number, and all lepton numbers to zero.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the resulting metadata violates
    /// another particle invariant.
    pub fn with_zero_additive_quantum_numbers(mut self) -> LadduPhysicsResult<Self> {
        self.charge = Some(0);
        self.strangeness = Some(0);
        self.charm = Some(0);
        self.bottomness = Some(0);
        self.topness = Some(0);
        self.baryon_number = Some(0);
        self.electron_lepton_number = Some(0);
        self.muon_lepton_number = Some(0);
        self.tau_lepton_number = Some(0);

        self.check_invariants()?;
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

    /// External identifiers for the particle
    pub fn ids(&self) -> &IndexMap<String, ExternalId> {
        &self.ids
    }

    /// Return the first external identifier in the requested namespace.
    pub fn id(&self, namespace: &str) -> Option<&ExternalId> {
        self.ids.get(namespace)
    }

    /// Append an external identifier.
    pub fn with_id<Id: Into<ExternalId>>(mut self, namespace: &str, id: Id) -> Self {
        self.ids.insert(namespace.to_string(), id.into());
        self
    }

    /// Replace external identifiers.
    pub fn with_ids<I, S, Id>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = (S, Id)>,
        S: AsRef<str>,
        Id: Into<ExternalId>,
    {
        self.ids = ids
            .into_iter()
            .map(|(s, id)| (s.as_ref().to_string(), id.into()))
            .collect();
        self
    }
}

impl ParticleProperties {
    fn additive_quantum_number_fields(&self) -> [(&'static str, Option<i32>); 9] {
        [
            ("charge", self.charge),
            ("strangeness", self.strangeness),
            ("charm", self.charm),
            ("bottomness", self.bottomness),
            ("topness", self.topness),
            ("baryon_number", self.baryon_number),
            ("electron_lepton_number", self.electron_lepton_number),
            ("muon_lepton_number", self.muon_lepton_number),
            ("tau_lepton_number", self.tau_lepton_number),
        ]
    }

    fn fill_zero_additive_qns_if_self_conjugate(&mut self) {
        if self.self_conjugate == Some(true) {
            self.charge.get_or_insert(0);
            self.strangeness.get_or_insert(0);
            self.charm.get_or_insert(0);
            self.bottomness.get_or_insert(0);
            self.topness.get_or_insert(0);
            self.baryon_number.get_or_insert(0);
            self.electron_lepton_number.get_or_insert(0);
            self.muon_lepton_number.get_or_insert(0);
            self.tau_lepton_number.get_or_insert(0);
        }
    }

    fn check_self_conjugate_additive_qns(&self) -> LadduPhysicsResult<()> {
        if self.self_conjugate == Some(true) {
            for (property, value) in self.additive_quantum_number_fields() {
                if matches!(value, Some(v) if v != 0) {
                    return Err(LadduPhysicsError::invalid_relation(format!(
                        "self-conjugate particles must have {property} = 0"
                    )));
                }
            }
        }
        Ok(())
    }

    fn check_c_parity_allowed(&self) -> LadduPhysicsResult<()> {
        if self.c_parity.is_some() && self.self_conjugate == Some(false) {
            return Err(LadduPhysicsError::invalid_relation(
                "C-parity is only applicable to self-conjugate particles",
            ));
        }
        Ok(())
    }

    fn check_invariants(&self) -> LadduPhysicsResult<()> {
        self.check_self_conjugate_additive_qns()?;
        self.check_c_parity_allowed()?;
        Ok(())
    }

    fn apply_update(mut self, update: ParticleUpdate) -> LadduPhysicsResult<Self> {
        match update {
            ParticleUpdate::Species(species) => {
                if self.self_conjugate == Some(true) {
                    match &self.antiparticle_species {
                        Some(anti) if anti != &species => {
                            return Err(LadduPhysicsError::invalid_relation(
                                "self-conjugate particle cannot have distinct species and antiparticle_species",
                            ));
                        }
                        None => self.antiparticle_species = Some(species.clone()),
                        _ => {}
                    }
                }
                self.species = Some(species);
            }
            ParticleUpdate::AntiparticleSpecies(antiparticle_species) => {
                if self.self_conjugate == Some(true) {
                    match &self.species {
                        Some(species) if species != &antiparticle_species => {
                            return Err(LadduPhysicsError::invalid_relation(
                                "self-conjugate particle cannot have distinct species and antiparticle_species",
                            ));
                        }
                        None => self.species = Some(antiparticle_species.clone()),
                        _ => {}
                    }
                }
                self.antiparticle_species = Some(antiparticle_species);
            }
            ParticleUpdate::SpeciesNames(species, antiparticle_species) => {
                self.self_conjugate = Some(species == antiparticle_species);
                self.species = Some(species);
                self.antiparticle_species = Some(antiparticle_species);
                self.fill_zero_additive_qns_if_self_conjugate();
            }
            ParticleUpdate::SelfConjugate(value) => {
                if value {
                    if let (Some(species), Some(anti)) = (&self.species, &self.antiparticle_species)
                        && species != anti
                    {
                        return Err(LadduPhysicsError::invalid_relation(
                            "self-conjugate particle cannot have distinct species and antiparticle_species",
                        ));
                    }
                    match (&self.species, &self.antiparticle_species) {
                        (Some(species), None) => {
                            self.antiparticle_species = Some(species.clone());
                        }
                        (None, Some(anti)) => self.species = Some(anti.clone()),
                        _ => {}
                    }
                    self.self_conjugate = Some(true);
                    self.fill_zero_additive_qns_if_self_conjugate();
                } else {
                    if self.c_parity.is_some() {
                        return Err(LadduPhysicsError::invalid_relation(
                            "non-self-conjugate particles cannot have C-parity",
                        ));
                    }
                    self.self_conjugate = Some(false);
                }
            }
            ParticleUpdate::SelfConjugateSpecies(species) => {
                self.species = Some(species.clone());
                self.antiparticle_species = Some(species);
                self.self_conjugate = Some(true);
                self.fill_zero_additive_qns_if_self_conjugate();
            }
            ParticleUpdate::CParity(c) => {
                if self.self_conjugate == Some(false) {
                    return Err(LadduPhysicsError::invalid_relation(
                        "C-parity is only applicable to self-conjugate particles",
                    ));
                }
                self.c_parity = Some(c);
                if self.self_conjugate.is_none() {
                    self = self.apply_update(ParticleUpdate::SelfConjugate(true))?;
                }
            }
            ParticleUpdate::Additive(property, value) => {
                if self.self_conjugate == Some(true) && value != 0 {
                    return Err(LadduPhysicsError::invalid_value(
                        property.name(),
                        "0 for self-conjugate particles",
                        value,
                    ));
                }
                *property.field(&mut self) = Some(value);
            }
            ParticleUpdate::Statistics(statistics) => {
                if let Some(spin) = self.spin
                    && Statistics::from_spin(spin) != statistics
                {
                    return Err(LadduPhysicsError::invalid_relation(
                        "spin and statistics must be consistent",
                    ));
                }
                self.statistics = Some(statistics);
                return Ok(self);
            }
        }

        self.check_invariants()?;
        Ok(self)
    }
}

fn validate_projection(spin: J, projection: M) -> LadduPhysicsResult<()> {
    if projection.doubled().unsigned_abs() > spin.doubled() {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "spin projection must satisfy -J <= m <= J, got J = {spin}, m = {projection}"
        )));
    }
    if !spin.has_same_parity_as(projection) {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "spin projection must have the same integer/half-integer parity as spin, got J = {spin}, m = {projection}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ExternalId, ParticleProperties, ParticlePropertiesPatch};
    use crate::quantum::{J, Parity, Statistics};

    #[test]
    fn particle_properties_store_external_ids() {
        let properties = ParticleProperties::unknown()
            .with_id("pdg", ExternalId::code(310))
            .with_id("gluex", ExternalId::label("ks-short"));

        assert_eq!(properties.ids().len(), 2);
        assert_eq!(
            properties.id("pdg").and_then(ExternalId::code_value),
            Some(310)
        );
        assert_eq!(
            properties.id("gluex").and_then(ExternalId::label_value),
            Some("ks-short")
        );
        assert_eq!(properties.id("missing"), None);

        let replaced = properties.with_ids([("geant", ExternalId::code(16))]);
        assert_eq!(replaced.ids().len(), 1);
        assert_eq!(
            replaced.id("geant").and_then(ExternalId::code_value),
            Some(16)
        );
    }

    #[test]
    fn self_conjugate_identity_updates_are_order_independent() {
        let species_first = ParticleProperties::unknown()
            .with_species("gamma")
            .unwrap()
            .with_self_conjugate(true)
            .unwrap();
        let conjugacy_first = ParticleProperties::unknown()
            .with_self_conjugate(true)
            .unwrap()
            .with_species("gamma")
            .unwrap();

        assert_eq!(species_first, conjugacy_first);
        assert_eq!(species_first.species.as_deref(), Some("gamma"));
        assert_eq!(species_first.antiparticle_species.as_deref(), Some("gamma"));
    }

    #[test]
    fn bulk_update_checks_every_self_conjugate_additive_quantum_number() {
        let cases = [
            (
                "charge",
                ParticlePropertiesPatch {
                    charge: Some(1),
                    ..Default::default()
                },
            ),
            (
                "strangeness",
                ParticlePropertiesPatch {
                    strangeness: Some(1),
                    ..Default::default()
                },
            ),
            (
                "charm",
                ParticlePropertiesPatch {
                    charm: Some(1),
                    ..Default::default()
                },
            ),
            (
                "bottomness",
                ParticlePropertiesPatch {
                    bottomness: Some(1),
                    ..Default::default()
                },
            ),
            (
                "topness",
                ParticlePropertiesPatch {
                    topness: Some(1),
                    ..Default::default()
                },
            ),
            (
                "baryon_number",
                ParticlePropertiesPatch {
                    baryon_number: Some(1),
                    ..Default::default()
                },
            ),
            (
                "electron_lepton_number",
                ParticlePropertiesPatch {
                    electron_lepton_number: Some(1),
                    ..Default::default()
                },
            ),
            (
                "muon_lepton_number",
                ParticlePropertiesPatch {
                    muon_lepton_number: Some(1),
                    ..Default::default()
                },
            ),
            (
                "tau_lepton_number",
                ParticlePropertiesPatch {
                    tau_lepton_number: Some(1),
                    ..Default::default()
                },
            ),
        ];

        for (property, patch) in cases {
            let error = ParticleProperties::unknown()
                .with_self_conjugate(true)
                .unwrap()
                .apply_patch(patch)
                .unwrap_err();
            assert!(error.to_string().contains(property));
        }
    }

    #[test]
    fn bulk_update_validates_spin_statistics_and_normalizes_c_parity() {
        let mismatch = ParticleProperties::unknown().apply_patch(ParticlePropertiesPatch {
            spin: Some(J::int(1)),
            statistics: Some(Statistics::Fermion),
            ..Default::default()
        });
        assert_eq!(
            mismatch.unwrap_err().to_string(),
            "Invalid relation: spin and statistics must be consistent"
        );

        let particle = ParticleProperties::unknown()
            .apply_patch(ParticlePropertiesPatch {
                species: Some("pi0".into()),
                c_parity: Some(Parity::Positive),
                spin: Some(J::int(0)),
                statistics: Some(Statistics::Boson),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(particle.self_conjugate, Some(true));
        assert_eq!(particle.antiparticle_species.as_deref(), Some("pi0"));
        assert!(
            particle
                .additive_quantum_number_fields()
                .into_iter()
                .all(|(_, value)| value == Some(0))
        );

        let invalid_mass = ParticleProperties::unknown().apply_patch(ParticlePropertiesPatch {
            mass: Some(f64::NAN),
            ..Default::default()
        });
        assert!(invalid_mass.is_err());
    }
}
