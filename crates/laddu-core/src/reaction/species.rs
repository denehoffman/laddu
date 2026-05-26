use serde::{Deserialize, Serialize};

use crate::{
    quantum::{Charge, Isospin, Parity, ParticleProperties, Statistics, J},
    LadduError, LadduResult,
};

/// An external identifier associated with a physical particle species.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ExternalId {
    /// A numeric identifier in a named namespace, such as a PDG code.
    Code {
        /// Identifier namespace.
        namespace: String,
        /// Identifier value.
        value: i64,
    },
    /// A textual identifier in a named namespace.
    Label {
        /// Identifier namespace.
        namespace: String,
        /// Identifier value.
        value: String,
    },
}

impl ExternalId {
    /// Construct a numeric identifier in an arbitrary namespace.
    pub fn new(namespace: impl Into<String>, value: i64) -> Self {
        Self::Code {
            namespace: namespace.into(),
            value,
        }
    }

    /// Construct a PDG identifier.
    pub fn pdg(value: i64) -> Self {
        Self::new("pdg", value)
    }

    /// Construct a textual identifier in an arbitrary namespace.
    pub fn label(namespace: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Label {
            namespace: namespace.into(),
            value: value.into(),
        }
    }
}

/// Reusable physical identity and quantum-number information for channel particles.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Species {
    name: String,
    mass: Option<f64>,
    properties: ParticleProperties,
    ids: Vec<ExternalId>,
}

impl From<&Species> for Species {
    fn from(species: &Species) -> Self {
        species.clone()
    }
}

impl Species {
    /// Construct a species with a display name and no assumed physical properties.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            properties: ParticleProperties::unknown()
                .with_name(name.clone())
                .with_species(name.clone()),
            name,
            mass: None,
            ids: Vec::new(),
        }
    }

    /// Return the species name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the physical mass used for generation or validation.
    pub fn mass(mut self, mass: f64) -> LadduResult<Self> {
        if !mass.is_finite() || mass < 0.0 {
            return Err(LadduError::Custom(
                "species mass must be finite and non-negative".to_string(),
            ));
        }
        self.mass = Some(mass);
        Ok(self)
    }

    /// Return the configured physical mass, if one was supplied.
    pub const fn mass_value(&self) -> Option<f64> {
        self.mass
    }

    /// Set the species identity used by particle/antiparticle checks.
    ///
    /// The identity defaults to the name passed to [`Species::new`].
    pub fn species(mut self, species: impl Into<String>) -> Self {
        self.properties = self.properties.with_species(species);
        self
    }

    /// Set the identity of this species' antiparticle.
    pub fn antiparticle_species(mut self, antiparticle_species: impl Into<String>) -> Self {
        self.properties = self
            .properties
            .with_antiparticle_species(antiparticle_species);
        self
    }

    /// Set whether this species is its own antiparticle.
    pub fn self_conjugate(mut self, value: bool) -> Self {
        self.properties = self.properties.with_self_conjugate(value);
        self
    }

    /// Set the intrinsic spin, checking any already specified statistics.
    pub fn spin(mut self, spin: J) -> LadduResult<Self> {
        if let Some(statistics) = self.properties.statistics {
            if Statistics::from_spin(spin) != statistics {
                return Err(LadduError::Custom(
                    "spin and statistics must be consistent".to_string(),
                ));
            }
        }
        self.properties = self.properties.with_spin(spin);
        Ok(self)
    }

    /// Set the intrinsic parity.
    pub fn parity(mut self, parity: Parity) -> Self {
        self.properties = self.properties.with_parity(parity);
        self
    }

    /// Set the intrinsic C-parity.
    pub fn c_parity(mut self, parity: Parity) -> Self {
        self.properties = self.properties.with_c_parity(parity);
        self
    }

    /// Set the intrinsic G-parity.
    pub fn g_parity(mut self, parity: Parity) -> Self {
        self.properties = self.properties.with_g_parity(parity);
        self
    }

    /// Set electric charge.
    pub fn charge(mut self, charge: Charge) -> Self {
        self.properties = self.properties.with_charge(charge);
        self
    }

    /// Set isospin.
    pub fn isospin(mut self, isospin: Isospin) -> Self {
        self.properties = self.properties.with_isospin(isospin);
        self
    }

    /// Set total strangeness.
    pub fn strangeness(mut self, strangeness: i32) -> Self {
        self.properties = self.properties.with_strangeness(strangeness);
        self
    }

    /// Set total charm.
    pub fn charm(mut self, charm: i32) -> Self {
        self.properties = self.properties.with_charm(charm);
        self
    }

    /// Set total bottomness.
    pub fn bottomness(mut self, bottomness: i32) -> Self {
        self.properties = self.properties.with_bottomness(bottomness);
        self
    }

    /// Set total topness.
    pub fn topness(mut self, topness: i32) -> Self {
        self.properties = self.properties.with_topness(topness);
        self
    }

    /// Set total baryon number.
    pub fn baryon_number(mut self, baryon_number: i32) -> Self {
        self.properties = self.properties.with_baryon_number(baryon_number);
        self
    }

    /// Set electron lepton number.
    pub fn electron_lepton_number(mut self, value: i32) -> Self {
        self.properties = self.properties.with_electron_lepton_number(value);
        self
    }

    /// Set muon lepton number.
    pub fn muon_lepton_number(mut self, value: i32) -> Self {
        self.properties = self.properties.with_muon_lepton_number(value);
        self
    }

    /// Set tau lepton number.
    pub fn tau_lepton_number(mut self, value: i32) -> Self {
        self.properties = self.properties.with_tau_lepton_number(value);
        self
    }

    /// Set statistical nature, checking any already specified spin.
    pub fn statistics(mut self, statistics: Statistics) -> LadduResult<Self> {
        self.properties = self.properties.with_statistics(statistics)?;
        Ok(self)
    }

    /// Add an external identifier.
    pub fn id(mut self, id: ExternalId) -> Self {
        self.ids.push(id);
        self
    }

    /// Return all external identifiers assigned to this species.
    pub fn ids(&self) -> &[ExternalId] {
        &self.ids
    }

    /// Return the underlying quantum-number properties.
    pub const fn properties(&self) -> &ParticleProperties {
        &self.properties
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{j, m, quantum::Statistics, s};

    #[test]
    fn species_validates_mass_and_spin_statistics() {
        let species = Species::new("proton")
            .mass(0.938)
            .unwrap()
            .id(ExternalId::pdg(2212))
            .statistics(Statistics::Fermion)
            .unwrap()
            .spin(j!(1 / 2))
            .unwrap();
        assert_eq!(species.mass_value(), Some(0.938));
        assert_eq!(species.ids(), &[ExternalId::pdg(2212)]);
        assert_eq!(species.properties().species().unwrap(), "proton");
        assert!(Species::new("bad").mass(-1.0).is_err());
        assert!(species.spin(j!(0)).is_err());
    }

    #[test]
    fn species_sets_all_particle_properties() {
        let properties = Species::new("electron")
            .species("e-")
            .antiparticle_species("e+")
            .self_conjugate(false)
            .spin(s!(1 / 2))
            .unwrap()
            .parity(Parity::Positive)
            .c_parity(Parity::Negative)
            .g_parity(Parity::Positive)
            .charge(Charge::integer(-1))
            .isospin(Isospin::new(j!(1 / 2), Some(m!(-1 / 2))).unwrap())
            .strangeness(-1)
            .charm(1)
            .bottomness(-1)
            .topness(1)
            .baryon_number(1)
            .electron_lepton_number(1)
            .muon_lepton_number(-1)
            .tau_lepton_number(1)
            .statistics(Statistics::Fermion)
            .unwrap()
            .properties()
            .clone();

        assert_eq!(properties.name().unwrap(), "electron");
        assert_eq!(properties.species().unwrap(), "e-");
        assert_eq!(properties.antiparticle_species().unwrap(), "e+");
        assert!(!properties.self_conjugate().unwrap());
        assert_eq!(properties.spin().unwrap(), s!(1 / 2));
        assert_eq!(properties.parity().unwrap(), Parity::Positive);
        assert_eq!(properties.c_parity().unwrap(), Parity::Negative);
        assert_eq!(properties.g_parity().unwrap(), Parity::Positive);
        assert_eq!(properties.charge().unwrap(), Charge::integer(-1));
        assert_eq!(
            properties.isospin().unwrap(),
            Isospin::new(j!(1 / 2), Some(m!(-1 / 2))).unwrap()
        );
        assert_eq!(properties.strangeness().unwrap(), -1);
        assert_eq!(properties.charm().unwrap(), 1);
        assert_eq!(properties.bottomness().unwrap(), -1);
        assert_eq!(properties.topness().unwrap(), 1);
        assert_eq!(properties.baryon_number().unwrap(), 1);
        assert_eq!(properties.electron_lepton_number().unwrap(), 1);
        assert_eq!(properties.muon_lepton_number().unwrap(), -1);
        assert_eq!(properties.tau_lepton_number().unwrap(), 1);
        assert_eq!(properties.statistics().unwrap(), Statistics::Fermion);
    }
}
