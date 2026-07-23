use std::sync::LazyLock;

pub use baryons::*;
pub use fundamental::*;
pub use mesons::*;

use crate::{
    j,
    quantum::{Parity, ParticleProperties},
};

// TODO: verify all masses and properties with the pdg

/// Convenient constructions of fundamental particle properties.
///
/// # Note
///
/// Quarks are omitted as they have fractional charge and baryon number, and we do not actually gain
/// any use in being able to represent them in reactions which would outweigh the inconvenience of
/// representing these quantities as potentially fractional. Gluons are omitted because they also
/// cannot be observed as a color singlet.
pub mod fundamental {
    pub use bosons::*;
    pub use leptons::*;

    use super::*;

    /// Standard Model gauge and Higgs bosons.
    pub mod bosons {
        use super::*;
        use crate::l;

        /// Photon particle properties.
        pub static PHOTON: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::boson(l!(1))
                .with_name("gamma")
                .with_self_conjugate_species("photon")
                .unwrap()
                .with_parity(Parity::Negative)
                .with_c_parity(Parity::Negative)
                .unwrap()
                .with_mass(0.0)
                .with_id("pdg", 22)
            // NOTE: isospin not included since it can act as 0 or 1 depending on the circumstances
        });

        /// Positively charged W-boson particle properties.
        pub static W_PLUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::boson(l!(1))
                .with_name("W+")
                .with_species_names("W+", "W-")
                .unwrap()
                .with_charge(1)
                .with_isospin(j!(0).into())
                .with_mass(80.369)
                .with_id("pdg", 24)
        });

        /// Negatively charged W-boson particle properties.
        pub static W_MINUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::boson(l!(1))
                .with_name("W-")
                .with_species_names("W-", "W+")
                .unwrap()
                .with_charge(-1)
                .with_isospin(j!(0).into())
                .with_mass(80.369)
                .with_id("pdg", -24)
        });

        /// Z-boson particle properties.
        pub static Z_BOSON: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::boson(l!(1))
                .with_name("Z0")
                .with_self_conjugate_species("Z")
                .unwrap()
                .with_isospin(j!(0).into())
                .with_mass(91.188)
                .with_id("pdg", 23)
        });

        /// Higgs-boson particle properties.
        pub static HIGGS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::boson(l!(0))
                .with_name("H")
                .with_self_conjugate_species("Higgs")
                .unwrap()
                .with_parity(Parity::Positive)
                .with_c_parity(Parity::Positive)
                .unwrap()
                .with_isospin(j!(0).into())
                .with_mass(125.20)
                .with_id("pdg", 25)
        });
    }

    /// Charged leptons and neutrinos.
    pub mod leptons {
        use super::*;
        /// Electron particle properties.
        pub static ELECTRON: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(1, 0, 0)
                .with_name("e-")
                .with_species_names("electron", "positron")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(-1)
                .with_isospin(j!(0).into())
                .with_mass(0.000510998950)
                .with_id("pdg", 11)
        });

        /// Positron particle properties.
        pub static POSITRON: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(-1, 0, 0)
                .with_name("e+")
                .with_species_names("positron", "electron")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(1)
                .with_isospin(j!(0).into())
                .with_mass(0.000510998950)
                .with_id("pdg", -11)
        });

        /// Muon particle properties.
        pub static MUON: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(0, 1, 0)
                .with_name("mu-")
                .with_species_names("muon", "antimuon")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(-1)
                .with_isospin(j!(0).into())
                .with_mass(0.1056583755)
                .with_id("pdg", 13)
        });

        /// Antimuon particle properties.
        pub static ANTIMUON: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(0, -1, 0)
                .with_name("mu+")
                .with_species_names("antimuon", "muon")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(1)
                .with_isospin(j!(0).into())
                .with_mass(0.1056583755)
                .with_id("pdg", -13)
        });

        /// Tau-lepton particle properties.
        pub static TAU: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(0, 0, 1)
                .with_name("tau-")
                .with_species_names("tau", "antitau")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(-1)
                .with_isospin(j!(0).into())
                .with_mass(1.77693)
                .with_id("pdg", 15)
        });

        /// Antitau particle properties.
        pub static ANTITAU: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(0, 0, -1)
                .with_name("tau+")
                .with_species_names("antitau", "tau")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(1)
                .with_isospin(j!(0).into())
                .with_mass(1.77693)
                .with_id("pdg", -15)
        });

        /// Electron-neutrino particle properties.
        pub static ELECTRON_NEUTRINO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(1, 0, 0)
                .with_name("nu_e")
                .with_species_names("nu_e", "nubar_e")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(0)
                .with_isospin(j!(0).into())
                .with_id("pdg", 12)
        });

        /// Electron-antineutrino particle properties.
        pub static ELECTRON_ANTINEUTRINO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(-1, 0, 0)
                .with_name("nubar_e")
                .with_species_names("nubar_e", "nu_e")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(0)
                .with_isospin(j!(0).into())
                .with_id("pdg", -12)
        });

        /// Muon-neutrino particle properties.
        pub static MUON_NEUTRINO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(0, 1, 0)
                .with_name("nu_mu")
                .with_species_names("nu_mu", "nubar_mu")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(0)
                .with_isospin(j!(0).into())
                .with_id("pdg", 14)
        });

        /// Muon-antineutrino particle properties.
        pub static MUON_ANTINEUTRINO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(0, -1, 0)
                .with_name("nubar_mu")
                .with_species_names("nubar_mu", "nu_mu")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(0)
                .with_isospin(j!(0).into())
                .with_id("pdg", -14)
        });

        /// Tau-neutrino particle properties.
        pub static TAU_NEUTRINO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(0, 0, 1)
                .with_name("nu_tau")
                .with_species_names("nu_tau", "nubar_tau")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(0)
                .with_isospin(j!(0).into())
                .with_id("pdg", 16)
        });

        /// Tau-antineutrino particle properties.
        pub static TAU_ANTINEUTRINO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::lepton(0, 0, -1)
                .with_name("nu_tau_bar")
                .with_species_names("nubar_tau", "nu_tau")
                .unwrap()
                .with_spin(j!(1 / 2))
                .with_charge(0)
                .with_isospin(j!(0).into())
                .with_id("pdg", -16)
        });
    }
}

/// Built-in meson particle definitions.
pub mod mesons {
    pub use open_bottom::*;
    pub use open_charm::*;
    pub use pseudoscalars::*;
    pub use vectors::*;

    use super::*;

    /// Light and strange pseudoscalar mesons.
    pub mod pseudoscalars {
        use super::*;

        /// Positively charged pion particle properties.
        pub static PI_PLUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_zero_flavor()
                .with_name("pi+")
                .with_species_names("pi+", "pi-")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(1)
                .with_isospin(j!(1).into())
                .with_mass(0.13957039)
                .with_id("pdg", 211)
        });

        /// Negatively charged pion particle properties.
        pub static PI_MINUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_zero_flavor()
                .with_name("pi-")
                .with_species_names("pi-", "pi+")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(-1)
                .with_isospin(j!(1).into())
                .with_mass(0.13957039)
                .with_id("pdg", -211)
        });

        /// Neutral pion particle properties.
        pub static PI_ZERO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_zero_flavor()
                .with_name("pi0")
                .with_self_conjugate_species("pi0")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_c_parity(Parity::Positive)
                .unwrap()
                .with_g_parity(Parity::Negative)
                .with_charge(0)
                .with_isospin(j!(1).into())
                .with_mass(0.1349768)
                .with_id("pdg", 111)
        });

        /// Positively charged kaon particle properties.
        pub static K_PLUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("K+")
                .with_species_names("K+", "K-")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(1)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(1, 0, 0, 0)
                .unwrap()
                .with_mass(0.493677)
                .with_id("pdg", 321)
        });

        /// Negatively charged kaon particle properties.
        pub static K_MINUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("K-")
                .with_species_names("K-", "K+")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(-1)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(-1, 0, 0, 0)
                .unwrap()
                .with_mass(0.493677)
                .with_id("pdg", -321)
        });

        /// Neutral kaon particle properties.
        pub static K_ZERO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("K0")
                .with_species_names("K0", "K0_bar")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(0)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(1, 0, 0, 0)
                .unwrap()
                .with_mass(0.497611)
                .with_id("pdg", 311)
        });

        /// Neutral antikaon particle properties.
        pub static K_ZERO_BAR: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("K0_bar")
                .with_species_names("K0_bar", "K0")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(0)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(-1, 0, 0, 0)
                .unwrap()
                .with_mass(0.497611)
                .with_id("pdg", -311)
        });

        /// Short-lived neutral kaon particle properties.
        pub static K_SHORT: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("K_S0")
                .with_self_conjugate_species("K_S0")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_mass(0.497611)
                .with_id("pdg", 310)
        });

        /// Long-lived neutral kaon particle properties.
        pub static K_LONG: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("K_L0")
                .with_self_conjugate_species("K_L0")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_mass(0.497611)
                .with_id("pdg", 130)
        });

        /// Eta-meson particle properties.
        pub static ETA: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("eta")
                .with_self_conjugate_species("eta")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_c_parity(Parity::Positive)
                .unwrap()
                .with_g_parity(Parity::Positive)
                .with_isospin(j!(0).into())
                .with_mass(0.547862)
                .with_id("pdg", 221)
        });

        /// Eta-prime-meson particle properties.
        pub static ETA_PRIME: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("eta'")
                .with_self_conjugate_species("eta'")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_c_parity(Parity::Positive)
                .unwrap()
                .with_g_parity(Parity::Positive)
                .with_isospin(j!(0).into())
                .with_mass(0.95778)
                .with_id("pdg", 331)
        });
    }

    /// Light vector mesons and charmonium.
    pub mod vectors {
        use super::*;

        /// Positively charged rho-meson particle properties.
        pub static RHO_PLUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("rho+")
                .with_species_names("rho+", "rho-")
                .unwrap()
                .with_spin(j!(1))
                .with_parity(Parity::Negative)
                .with_charge(1)
                .with_isospin(j!(1).into())
                .with_mass(0.77511)
                .with_id("pdg", 213)
        });

        /// Negatively charged rho-meson particle properties.
        pub static RHO_MINUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("rho-")
                .with_species_names("rho-", "rho+")
                .unwrap()
                .with_spin(j!(1))
                .with_parity(Parity::Negative)
                .with_charge(-1)
                .with_isospin(j!(1).into())
                .with_mass(0.77511)
                .with_id("pdg", -213)
        });

        /// Neutral rho-meson particle properties.
        pub static RHO_ZERO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("rho0")
                .with_self_conjugate_species("rho0")
                .unwrap()
                .with_spin(j!(1))
                .with_parity(Parity::Negative)
                .with_c_parity(Parity::Negative)
                .unwrap()
                .with_g_parity(Parity::Positive)
                .with_isospin(j!(1).into())
                .with_mass(0.77511)
                .with_id("pdg", 113)
        });

        /// Omega-meson particle properties.
        pub static OMEGA: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("omega")
                .with_self_conjugate_species("omega")
                .unwrap()
                .with_spin(j!(1))
                .with_parity(Parity::Negative)
                .with_c_parity(Parity::Negative)
                .unwrap()
                .with_g_parity(Parity::Negative)
                .with_isospin(j!(0).into())
                .with_mass(0.78266)
                .with_id("pdg", 223)
        });

        /// Phi-meson particle properties.
        pub static PHI: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("phi")
                .with_self_conjugate_species("phi")
                .unwrap()
                .with_spin(j!(1))
                .with_parity(Parity::Negative)
                .with_c_parity(Parity::Negative)
                .unwrap()
                .with_g_parity(Parity::Negative)
                .with_isospin(j!(0).into())
                .with_mass(1.019461)
                .with_id("pdg", 333)
        });

        /// J/psi-meson particle properties.
        pub static J_PSI: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("J/psi")
                .with_self_conjugate_species("J/psi")
                .unwrap()
                .with_spin(j!(1))
                .with_parity(Parity::Negative)
                .with_c_parity(Parity::Negative)
                .unwrap()
                .with_isospin(j!(0).into())
                .with_mass(3.0969)
                .with_id("pdg", 443)
        });
    }

    /// Open-charm pseudoscalar mesons.
    pub mod open_charm {
        use super::*;

        /// Positively charged D-meson particle properties.
        pub static D_PLUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("D+")
                .with_species_names("D+", "D-")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(1)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(0, 1, 0, 0)
                .unwrap()
                .with_mass(1.86966)
                .with_id("pdg", 411)
        });

        /// Negatively charged D-meson particle properties.
        pub static D_MINUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("D-")
                .with_species_names("D-", "D+")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(-1)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(0, -1, 0, 0)
                .unwrap()
                .with_mass(1.86966)
                .with_id("pdg", -411)
        });

        /// Neutral D-meson particle properties.
        pub static D_ZERO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("D0")
                .with_species_names("D0", "D0_bar")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(0)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(0, 1, 0, 0)
                .unwrap()
                .with_mass(1.86484)
                .with_id("pdg", 421)
        });

        /// Neutral anti-D-meson particle properties.
        pub static D_ZERO_BAR: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("D0_bar")
                .with_species_names("D0_bar", "D0")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(0)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(0, -1, 0, 0)
                .unwrap()
                .with_mass(1.86484)
                .with_id("pdg", -421)
        });

        /// Positively charged D-s-meson particle properties.
        pub static D_S_PLUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("D_s+")
                .with_species_names("D_s+", "D_s-")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(1)
                .with_isospin(j!(0).into())
                .with_flavor(1, 1, 0, 0)
                .unwrap()
                .with_mass(1.96835)
                .with_id("pdg", 431)
        });

        /// Negatively charged D-s-meson particle properties.
        pub static D_S_MINUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("D_s-")
                .with_species_names("D_s-", "D_s+")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(-1)
                .with_isospin(j!(0).into())
                .with_flavor(-1, -1, 0, 0)
                .unwrap()
                .with_mass(1.96835)
                .with_id("pdg", -431)
        });
    }

    /// Open-bottom pseudoscalar mesons.
    pub mod open_bottom {
        use super::*;

        /// Positively charged B-meson particle properties.
        pub static B_PLUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("B+")
                .with_species_names("B+", "B-")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(1)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(0, 0, 1, 0)
                .unwrap()
                .with_mass(5.27941)
                .with_id("pdg", 521)
        });

        /// Negatively charged B-meson particle properties.
        pub static B_MINUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("B-")
                .with_species_names("B-", "B+")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(-1)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(0, 0, -1, 0)
                .unwrap()
                .with_mass(5.27941)
                .with_id("pdg", -521)
        });

        /// Neutral B-meson particle properties.
        pub static B_ZERO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("B0")
                .with_species_names("B0", "B0_bar")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(0)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(0, 0, 1, 0)
                .unwrap()
                .with_mass(5.27972)
                .with_id("pdg", 511)
        });

        /// Neutral anti-B-meson particle properties.
        pub static B_ZERO_BAR: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("B0_bar")
                .with_species_names("B0_bar", "B0")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(0)
                .with_isospin(j!(1 / 2).into())
                .with_flavor(0, 0, -1, 0)
                .unwrap()
                .with_mass(5.27972)
                .with_id("pdg", -511)
        });

        /// Neutral B-s-meson particle properties.
        pub static B_S_ZERO: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("B_s0")
                .with_species_names("B_s0", "B_s0_bar")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(0)
                .with_isospin(j!(0).into())
                .with_flavor(-1, 0, 1, 0)
                .unwrap()
                .with_mass(5.36693)
                .with_id("pdg", 531)
        });

        /// Neutral anti-B-s-meson particle properties.
        pub static B_S_ZERO_BAR: LazyLock<ParticleProperties> = LazyLock::new(|| {
            ParticleProperties::meson()
                .with_name("B_s0_bar")
                .with_species_names("B_s0_bar", "B_s0")
                .unwrap()
                .with_spin(j!(0))
                .with_parity(Parity::Negative)
                .with_charge(0)
                .with_isospin(j!(0).into())
                .with_flavor(1, 0, -1, 0)
                .unwrap()
                .with_mass(5.36693)
                .with_id("pdg", -531)
        });
    }
}

/// Built-in baryon and antibaryon particle definitions.
pub mod baryons {
    use super::*;

    /// Proton particle properties.
    pub static PROTON: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(1)
            .with_zero_flavor()
            .with_name("p")
            .with_species_names("proton", "antiproton")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(1)
            .with_isospin(j!(1 / 2).into())
            .with_mass(0.93827208816)
            .with_id("pdg", 2212)
    });

    /// Antiproton particle properties.
    pub static ANTIPROTON: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(-1)
            .with_zero_flavor()
            .with_name("p_bar")
            .with_species_names("antiproton", "proton")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(-1)
            .with_isospin(j!(1 / 2).into())
            .with_mass(0.93827208816)
            .with_id("pdg", -2212)
    });

    /// Neutron particle properties.
    pub static NEUTRON: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(1)
            .with_zero_flavor()
            .with_name("n")
            .with_species_names("neutron", "antineutron")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(0)
            .with_isospin(j!(1 / 2).into())
            .with_mass(0.93956542052)
            .with_id("pdg", 2112)
    });

    /// Antineutron particle properties.
    pub static ANTINEUTRON: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(-1)
            .with_zero_flavor()
            .with_name("n_bar")
            .with_species_names("antineutron", "neutron")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(0)
            .with_isospin(j!(1 / 2).into())
            .with_mass(0.93956542052)
            .with_id("pdg", -2112)
    });

    /// Lambda-baryon particle properties.
    pub static LAMBDA: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(1)
            .with_name("Lambda0")
            .with_species_names("Lambda0", "Lambda0_bar")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(0)
            .with_isospin(j!(0).into())
            .with_flavor(-1, 0, 0, 0)
            .unwrap()
            .with_mass(1.115683)
            .with_id("pdg", 3122)
    });

    /// Anti-Lambda-baryon particle properties.
    pub static ANTILAMBDA: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(-1)
            .with_name("Lambda0_bar")
            .with_species_names("Lambda0_bar", "Lambda0")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(0)
            .with_isospin(j!(0).into())
            .with_flavor(1, 0, 0, 0)
            .unwrap()
            .with_mass(1.115683)
            .with_id("pdg", -3122)
    });

    /// Positively charged Sigma-baryon particle properties.
    pub static SIGMA_PLUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(1)
            .with_name("Sigma+")
            .with_species_names("Sigma+", "Sigma-_bar")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(1)
            .with_isospin(j!(1).into())
            .with_flavor(-1, 0, 0, 0)
            .unwrap()
            .with_mass(1.18937)
            .with_id("pdg", 3222)
    });

    /// Negatively charged anti-Sigma-baryon particle properties.
    pub static ANTISIGMA_MINUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(-1)
            .with_name("Sigma-_bar")
            .with_species_names("Sigma-_bar", "Sigma+")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(-1)
            .with_isospin(j!(1).into())
            .with_flavor(1, 0, 0, 0)
            .unwrap()
            .with_mass(1.18937)
            .with_id("pdg", -3222)
    });

    /// Neutral Sigma-baryon particle properties.
    pub static SIGMA_ZERO: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(1)
            .with_name("Sigma0")
            .with_species_names("Sigma0", "Sigma0_bar")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(0)
            .with_isospin(j!(1).into())
            .with_flavor(-1, 0, 0, 0)
            .unwrap()
            .with_mass(1.192642)
            .with_id("pdg", 3212)
    });

    /// Neutral anti-Sigma-baryon particle properties.
    pub static ANTISIGMA_ZERO: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(-1)
            .with_name("Sigma0_bar")
            .with_species_names("Sigma0_bar", "Sigma0")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(0)
            .with_isospin(j!(1).into())
            .with_flavor(1, 0, 0, 0)
            .unwrap()
            .with_mass(1.192642)
            .with_id("pdg", -3212)
    });

    /// Negatively charged Sigma-baryon particle properties.
    pub static SIGMA_MINUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(1)
            .with_name("Sigma-")
            .with_species_names("Sigma-", "Sigma+_bar")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(-1)
            .with_isospin(j!(1).into())
            .with_flavor(-1, 0, 0, 0)
            .unwrap()
            .with_mass(1.197449)
            .with_id("pdg", 3112)
    });

    /// Positively charged anti-Sigma-baryon particle properties.
    pub static ANTISIGMA_PLUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(-1)
            .with_name("Sigma+_bar")
            .with_species_names("Sigma+_bar", "Sigma-")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(1)
            .with_isospin(j!(1).into())
            .with_flavor(1, 0, 0, 0)
            .unwrap()
            .with_mass(1.197449)
            .with_id("pdg", -3112)
    });

    /// Neutral Xi-baryon particle properties.
    pub static XI_ZERO: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(1)
            .with_name("Xi0")
            .with_species_names("Xi0", "Xi0_bar")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(0)
            .with_isospin(j!(1 / 2).into())
            .with_flavor(-2, 0, 0, 0)
            .unwrap()
            .with_mass(1.31486)
            .with_id("pdg", 3322)
    });

    /// Neutral anti-Xi-baryon particle properties.
    pub static ANTI_XI_ZERO: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(-1)
            .with_name("Xi0_bar")
            .with_species_names("Xi0_bar", "Xi0")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(0)
            .with_isospin(j!(1 / 2).into())
            .with_flavor(2, 0, 0, 0)
            .unwrap()
            .with_mass(1.31486)
            .with_id("pdg", -3322)
    });

    /// Negatively charged Xi-baryon particle properties.
    pub static XI_MINUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(1)
            .with_name("Xi-")
            .with_species_names("Xi-", "Xi+_bar")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(-1)
            .with_isospin(j!(1 / 2).into())
            .with_flavor(-2, 0, 0, 0)
            .unwrap()
            .with_mass(1.32171)
            .with_id("pdg", 3312)
    });

    /// Positively charged anti-Xi-baryon particle properties.
    pub static ANTI_XI_PLUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(-1)
            .with_name("Xi+_bar")
            .with_species_names("Xi+_bar", "Xi-")
            .unwrap()
            .with_spin(j!(1 / 2))
            .with_parity(Parity::Positive)
            .with_charge(1)
            .with_isospin(j!(1 / 2).into())
            .with_flavor(2, 0, 0, 0)
            .unwrap()
            .with_mass(1.32171)
            .with_id("pdg", -3312)
    });

    /// Negatively charged Omega-baryon particle properties.
    pub static OMEGA_MINUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(1)
            .with_name("Omega-")
            .with_species_names("Omega-", "Omega+_bar")
            .unwrap()
            .with_spin(j!(3 / 2))
            .with_parity(Parity::Positive)
            .with_charge(-1)
            .with_isospin(j!(0).into())
            .with_flavor(-3, 0, 0, 0)
            .unwrap()
            .with_mass(1.67245)
            .with_id("pdg", 3334)
    });

    /// Positively charged anti-Omega-baryon particle properties.
    pub static ANTI_OMEGA_PLUS: LazyLock<ParticleProperties> = LazyLock::new(|| {
        ParticleProperties::baryon(-1)
            .with_name("Omega+_bar")
            .with_species_names("Omega+_bar", "Omega-")
            .unwrap()
            .with_spin(j!(3 / 2))
            .with_parity(Parity::Positive)
            .with_charge(1)
            .with_isospin(j!(0).into())
            .with_flavor(3, 0, 0, 0)
            .unwrap()
            .with_mass(1.67245)
            .with_id("pdg", -3334)
    });
}
