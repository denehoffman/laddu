use std::sync::LazyLock;

pub use baryons::*;
pub use fundamental::*;
pub use mesons::*;

use crate::quantum::{J, L, Parity, ParticleProperties};

// Audit reference for the compatibility-preserved values below; this conversion intentionally
// retains the catalog's existing precision and does not silently adopt newer central values.
const PDG_2024: &str = "S. Navas et al. (Particle Data Group), Phys. Rev. D 110, 030001 (2024)";

#[derive(Clone, Copy)]
enum ParticleKind {
    Boson,
    Lepton([i32; 3]),
    Meson,
    Baryon(i32),
}

#[derive(Clone, Copy)]
enum SpeciesRecord {
    Pair(&'static str, &'static str),
    SelfConjugate(&'static str),
}

#[derive(Clone, Copy)]
struct ParticleRecord {
    kind: ParticleKind,
    name: &'static str,
    species: SpeciesRecord,
    spin_twice: u32,
    parity: Option<Parity>,
    c_parity: Option<Parity>,
    g_parity: Option<Parity>,
    charge: Option<i32>,
    isospin_twice: Option<u32>,
    flavor: Option<[i32; 4]>,
    mass: Option<f64>,
    pdg_id: i64,
    provenance: &'static str,
}

impl ParticleRecord {
    const fn new(
        kind: ParticleKind,
        name: &'static str,
        species: SpeciesRecord,
        spin_twice: u32,
        pdg_id: i64,
    ) -> Self {
        Self {
            kind,
            name,
            species,
            spin_twice,
            parity: None,
            c_parity: None,
            g_parity: None,
            charge: None,
            isospin_twice: None,
            flavor: None,
            mass: None,
            pdg_id,
            provenance: PDG_2024,
        }
    }

    const fn parity(mut self, parity: Parity) -> Self {
        self.parity = Some(parity);
        self
    }

    const fn c_parity(mut self, parity: Parity) -> Self {
        self.c_parity = Some(parity);
        self
    }

    const fn g_parity(mut self, parity: Parity) -> Self {
        self.g_parity = Some(parity);
        self
    }

    const fn charge(mut self, charge: i32) -> Self {
        self.charge = Some(charge);
        self
    }

    const fn isospin(mut self, isospin_twice: u32) -> Self {
        self.isospin_twice = Some(isospin_twice);
        self
    }

    const fn flavor(mut self, strangeness: i32, charm: i32, bottomness: i32) -> Self {
        self.flavor = Some([strangeness, charm, bottomness, 0]);
        self
    }

    const fn mass(mut self, mass: f64) -> Self {
        self.mass = Some(mass);
        self
    }

    const fn antiparticle(mut self, name: &'static str) -> Self {
        self.name = name;
        self.species = match self.species {
            SpeciesRecord::Pair(particle, antiparticle) => {
                SpeciesRecord::Pair(antiparticle, particle)
            }
            SpeciesRecord::SelfConjugate(_) => {
                panic!("self-conjugate records do not have distinct antiparticles")
            }
        };
        self.kind = match self.kind {
            ParticleKind::Boson => ParticleKind::Boson,
            ParticleKind::Lepton([electron, muon, tau]) => {
                ParticleKind::Lepton([-electron, -muon, -tau])
            }
            ParticleKind::Meson => ParticleKind::Meson,
            ParticleKind::Baryon(number) => ParticleKind::Baryon(-number),
        };
        self.charge = match self.charge {
            Some(value) => Some(-value),
            None => None,
        };
        self.flavor = match self.flavor {
            Some([strangeness, charm, bottomness, topness]) => {
                Some([-strangeness, -charm, -bottomness, -topness])
            }
            None => None,
        };
        self.pdg_id = -self.pdg_id;
        self
    }

    fn build(self) -> ParticleProperties {
        let mut particle = match self.kind {
            ParticleKind::Boson => ParticleProperties::boson(L::int(self.spin_twice / 2)),
            ParticleKind::Lepton([electron, muon, tau]) => {
                ParticleProperties::lepton(electron, muon, tau)
            }
            ParticleKind::Meson => ParticleProperties::meson(),
            ParticleKind::Baryon(number) => ParticleProperties::baryon(number),
        }
        .with_name(self.name)
        .with_spin(J::half(self.spin_twice));

        match self.species {
            SpeciesRecord::Pair(species, antiparticle) => {
                particle.species = Some(species.into());
                particle.antiparticle_species = Some(antiparticle.into());
                particle.self_conjugate = Some(false);
            }
            SpeciesRecord::SelfConjugate(species) => {
                particle.species = Some(species.into());
                particle.antiparticle_species = Some(species.into());
                particle.self_conjugate = Some(true);
            }
        }
        particle.parity = self.parity;
        particle.c_parity = self.c_parity;
        particle.g_parity = self.g_parity;
        particle.charge = self.charge;
        particle.isospin = self.isospin_twice.map(|value| J::half(value).into());
        if let Some([strangeness, charm, bottomness, topness]) = self.flavor {
            particle.strangeness = Some(strangeness);
            particle.charm = Some(charm);
            particle.bottomness = Some(bottomness);
            particle.topness = Some(topness);
        }
        particle.mass = self.mass;
        particle = particle.with_id("pdg", self.pdg_id);
        particle.check_invariants().unwrap_or_else(|error| {
            panic!(
                "invalid builtin particle record from {}: {error}",
                self.provenance
            )
        });
        particle
    }
}

macro_rules! builtins {
    ($($(#[$meta:meta])* $name:ident => $record:expr;)+) => {
        $(
            $(#[$meta])*
            pub static $name: LazyLock<ParticleProperties> =
                LazyLock::new(|| ($record).build());
        )+
    };
}

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

        const W: ParticleRecord = ParticleRecord::new(
            ParticleKind::Boson,
            "W+",
            SpeciesRecord::Pair("W+", "W-"),
            2,
            24,
        )
        .charge(1)
        .isospin(0)
        .mass(80.369);

        builtins! {
            /// Photon particle properties.
            PHOTON => ParticleRecord::new(
                ParticleKind::Boson, "gamma", SpeciesRecord::SelfConjugate("photon"), 2, 22,
            ).parity(Parity::Negative).c_parity(Parity::Negative).mass(0.0);
            /// Positively charged W-boson particle properties.
            W_PLUS => W;
            /// Negatively charged W-boson particle properties.
            W_MINUS => W.antiparticle("W-");
            /// Z-boson particle properties.
            Z_BOSON => ParticleRecord::new(
                ParticleKind::Boson, "Z0", SpeciesRecord::SelfConjugate("Z"), 2, 23,
            ).isospin(0).mass(91.188);
            /// Higgs-boson particle properties.
            HIGGS => ParticleRecord::new(
                ParticleKind::Boson, "H", SpeciesRecord::SelfConjugate("Higgs"), 0, 25,
            ).parity(Parity::Positive).c_parity(Parity::Positive).isospin(0).mass(125.20);
        }
    }

    /// Charged leptons and neutrinos.
    pub mod leptons {
        use super::*;

        const ELECTRON_RECORD: ParticleRecord = ParticleRecord::new(
            ParticleKind::Lepton([1, 0, 0]),
            "e-",
            SpeciesRecord::Pair("electron", "positron"),
            1,
            11,
        )
        .charge(-1)
        .isospin(0)
        .mass(0.000510998950);
        const MUON_RECORD: ParticleRecord = ParticleRecord::new(
            ParticleKind::Lepton([0, 1, 0]),
            "mu-",
            SpeciesRecord::Pair("muon", "antimuon"),
            1,
            13,
        )
        .charge(-1)
        .isospin(0)
        .mass(0.1056583755);
        const TAU_RECORD: ParticleRecord = ParticleRecord::new(
            ParticleKind::Lepton([0, 0, 1]),
            "tau-",
            SpeciesRecord::Pair("tau", "antitau"),
            1,
            15,
        )
        .charge(-1)
        .isospin(0)
        .mass(1.77693);
        const ELECTRON_NEUTRINO_RECORD: ParticleRecord = ParticleRecord::new(
            ParticleKind::Lepton([1, 0, 0]),
            "nu_e",
            SpeciesRecord::Pair("nu_e", "nubar_e"),
            1,
            12,
        )
        .charge(0)
        .isospin(0);
        const MUON_NEUTRINO_RECORD: ParticleRecord = ParticleRecord::new(
            ParticleKind::Lepton([0, 1, 0]),
            "nu_mu",
            SpeciesRecord::Pair("nu_mu", "nubar_mu"),
            1,
            14,
        )
        .charge(0)
        .isospin(0);
        const TAU_NEUTRINO_RECORD: ParticleRecord = ParticleRecord::new(
            ParticleKind::Lepton([0, 0, 1]),
            "nu_tau",
            SpeciesRecord::Pair("nu_tau", "nubar_tau"),
            1,
            16,
        )
        .charge(0)
        .isospin(0);

        builtins! {
            /// Electron particle properties.
            ELECTRON => ELECTRON_RECORD;
            /// Positron particle properties.
            POSITRON => ELECTRON_RECORD.antiparticle("e+");
            /// Muon particle properties.
            MUON => MUON_RECORD;
            /// Antimuon particle properties.
            ANTIMUON => MUON_RECORD.antiparticle("mu+");
            /// Tau-lepton particle properties.
            TAU => TAU_RECORD;
            /// Antitau particle properties.
            ANTITAU => TAU_RECORD.antiparticle("tau+");
            /// Electron-neutrino particle properties.
            ELECTRON_NEUTRINO => ELECTRON_NEUTRINO_RECORD;
            /// Electron-antineutrino particle properties.
            ELECTRON_ANTINEUTRINO => ELECTRON_NEUTRINO_RECORD.antiparticle("nubar_e");
            /// Muon-neutrino particle properties.
            MUON_NEUTRINO => MUON_NEUTRINO_RECORD;
            /// Muon-antineutrino particle properties.
            MUON_ANTINEUTRINO => MUON_NEUTRINO_RECORD.antiparticle("nubar_mu");
            /// Tau-neutrino particle properties.
            TAU_NEUTRINO => TAU_NEUTRINO_RECORD;
            /// Tau-antineutrino particle properties.
            TAU_ANTINEUTRINO => TAU_NEUTRINO_RECORD.antiparticle("nu_tau_bar");
        }
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

        const PI: ParticleRecord = ParticleRecord::new(
            ParticleKind::Meson,
            "pi+",
            SpeciesRecord::Pair("pi+", "pi-"),
            0,
            211,
        )
        .parity(Parity::Negative)
        .charge(1)
        .isospin(2)
        .flavor(0, 0, 0)
        .mass(0.13957039);
        const K_CHARGED: ParticleRecord = ParticleRecord::new(
            ParticleKind::Meson,
            "K+",
            SpeciesRecord::Pair("K+", "K-"),
            0,
            321,
        )
        .parity(Parity::Negative)
        .charge(1)
        .isospin(1)
        .flavor(1, 0, 0)
        .mass(0.493677);
        const K_NEUTRAL: ParticleRecord = ParticleRecord::new(
            ParticleKind::Meson,
            "K0",
            SpeciesRecord::Pair("K0", "K0_bar"),
            0,
            311,
        )
        .parity(Parity::Negative)
        .charge(0)
        .isospin(1)
        .flavor(1, 0, 0)
        .mass(0.497611);

        builtins! {
            /// Positively charged pion particle properties.
            PI_PLUS => PI;
            /// Negatively charged pion particle properties.
            PI_MINUS => PI.antiparticle("pi-");
            /// Neutral pion particle properties.
            PI_ZERO => ParticleRecord::new(
                ParticleKind::Meson, "pi0", SpeciesRecord::SelfConjugate("pi0"), 0, 111,
            ).parity(Parity::Negative).c_parity(Parity::Positive).g_parity(Parity::Negative)
                .charge(0).isospin(2).flavor(0, 0, 0).mass(0.1349768);
            /// Positively charged kaon particle properties.
            K_PLUS => K_CHARGED;
            /// Negatively charged kaon particle properties.
            K_MINUS => K_CHARGED.antiparticle("K-");
            /// Neutral kaon particle properties.
            K_ZERO => K_NEUTRAL;
            /// Neutral antikaon particle properties.
            K_ZERO_BAR => K_NEUTRAL.antiparticle("K0_bar");
            /// Short-lived neutral kaon particle properties.
            K_SHORT => ParticleRecord::new(
                ParticleKind::Meson, "K_S0", SpeciesRecord::SelfConjugate("K_S0"), 0, 310,
            ).parity(Parity::Negative).mass(0.497611);
            /// Long-lived neutral kaon particle properties.
            K_LONG => ParticleRecord::new(
                ParticleKind::Meson, "K_L0", SpeciesRecord::SelfConjugate("K_L0"), 0, 130,
            ).parity(Parity::Negative).mass(0.497611);
            /// Eta-meson particle properties.
            ETA => ParticleRecord::new(
                ParticleKind::Meson, "eta", SpeciesRecord::SelfConjugate("eta"), 0, 221,
            ).parity(Parity::Negative).c_parity(Parity::Positive).g_parity(Parity::Positive)
                .isospin(0).mass(0.547862);
            /// Eta-prime-meson particle properties.
            ETA_PRIME => ParticleRecord::new(
                ParticleKind::Meson, "eta'", SpeciesRecord::SelfConjugate("eta'"), 0, 331,
            ).parity(Parity::Negative).c_parity(Parity::Positive).g_parity(Parity::Positive)
                .isospin(0).mass(0.95778);
        }
    }

    /// Light vector mesons and charmonium.
    pub mod vectors {
        use super::*;

        const RHO: ParticleRecord = ParticleRecord::new(
            ParticleKind::Meson,
            "rho+",
            SpeciesRecord::Pair("rho+", "rho-"),
            2,
            213,
        )
        .parity(Parity::Negative)
        .charge(1)
        .isospin(2)
        .mass(0.77511);

        builtins! {
            /// Positively charged rho-meson particle properties.
            RHO_PLUS => RHO;
            /// Negatively charged rho-meson particle properties.
            RHO_MINUS => RHO.antiparticle("rho-");
            /// Neutral rho-meson particle properties.
            RHO_ZERO => ParticleRecord::new(
                ParticleKind::Meson, "rho0", SpeciesRecord::SelfConjugate("rho0"), 2, 113,
            ).parity(Parity::Negative).c_parity(Parity::Negative).g_parity(Parity::Positive)
                .isospin(2).mass(0.77511);
            /// Omega-meson particle properties.
            OMEGA => ParticleRecord::new(
                ParticleKind::Meson, "omega", SpeciesRecord::SelfConjugate("omega"), 2, 223,
            ).parity(Parity::Negative).c_parity(Parity::Negative).g_parity(Parity::Negative)
                .isospin(0).mass(0.78266);
            /// Phi-meson particle properties.
            PHI => ParticleRecord::new(
                ParticleKind::Meson, "phi", SpeciesRecord::SelfConjugate("phi"), 2, 333,
            ).parity(Parity::Negative).c_parity(Parity::Negative).g_parity(Parity::Negative)
                .isospin(0).mass(1.019461);
            /// J/psi-meson particle properties.
            J_PSI => ParticleRecord::new(
                ParticleKind::Meson, "J/psi", SpeciesRecord::SelfConjugate("J/psi"), 2, 443,
            ).parity(Parity::Negative).c_parity(Parity::Negative).isospin(0).mass(3.0969);
        }
    }

    /// Open-charm pseudoscalar mesons.
    pub mod open_charm {
        use super::*;

        const D_CHARGED: ParticleRecord = ParticleRecord::new(
            ParticleKind::Meson,
            "D+",
            SpeciesRecord::Pair("D+", "D-"),
            0,
            411,
        )
        .parity(Parity::Negative)
        .charge(1)
        .isospin(1)
        .flavor(0, 1, 0)
        .mass(1.86966);
        const D_NEUTRAL: ParticleRecord = ParticleRecord::new(
            ParticleKind::Meson,
            "D0",
            SpeciesRecord::Pair("D0", "D0_bar"),
            0,
            421,
        )
        .parity(Parity::Negative)
        .charge(0)
        .isospin(1)
        .flavor(0, 1, 0)
        .mass(1.86484);
        const D_S: ParticleRecord = ParticleRecord::new(
            ParticleKind::Meson,
            "D_s+",
            SpeciesRecord::Pair("D_s+", "D_s-"),
            0,
            431,
        )
        .parity(Parity::Negative)
        .charge(1)
        .isospin(0)
        .flavor(1, 1, 0)
        .mass(1.96835);

        builtins! {
            /// Positively charged D-meson particle properties.
            D_PLUS => D_CHARGED;
            /// Negatively charged D-meson particle properties.
            D_MINUS => D_CHARGED.antiparticle("D-");
            /// Neutral D-meson particle properties.
            D_ZERO => D_NEUTRAL;
            /// Neutral anti-D-meson particle properties.
            D_ZERO_BAR => D_NEUTRAL.antiparticle("D0_bar");
            /// Positively charged D-s-meson particle properties.
            D_S_PLUS => D_S;
            /// Negatively charged D-s-meson particle properties.
            D_S_MINUS => D_S.antiparticle("D_s-");
        }
    }

    /// Open-bottom pseudoscalar mesons.
    pub mod open_bottom {
        use super::*;

        const B_CHARGED: ParticleRecord = ParticleRecord::new(
            ParticleKind::Meson,
            "B+",
            SpeciesRecord::Pair("B+", "B-"),
            0,
            521,
        )
        .parity(Parity::Negative)
        .charge(1)
        .isospin(1)
        .flavor(0, 0, 1)
        .mass(5.27941);
        const B_NEUTRAL: ParticleRecord = ParticleRecord::new(
            ParticleKind::Meson,
            "B0",
            SpeciesRecord::Pair("B0", "B0_bar"),
            0,
            511,
        )
        .parity(Parity::Negative)
        .charge(0)
        .isospin(1)
        .flavor(0, 0, 1)
        .mass(5.27972);
        const B_S: ParticleRecord = ParticleRecord::new(
            ParticleKind::Meson,
            "B_s0",
            SpeciesRecord::Pair("B_s0", "B_s0_bar"),
            0,
            531,
        )
        .parity(Parity::Negative)
        .charge(0)
        .isospin(0)
        .flavor(-1, 0, 1)
        .mass(5.36693);

        builtins! {
            /// Positively charged B-meson particle properties.
            B_PLUS => B_CHARGED;
            /// Negatively charged B-meson particle properties.
            B_MINUS => B_CHARGED.antiparticle("B-");
            /// Neutral B-meson particle properties.
            B_ZERO => B_NEUTRAL;
            /// Neutral anti-B-meson particle properties.
            B_ZERO_BAR => B_NEUTRAL.antiparticle("B0_bar");
            /// Neutral B-s-meson particle properties.
            B_S_ZERO => B_S;
            /// Neutral anti-B-s-meson particle properties.
            B_S_ZERO_BAR => B_S.antiparticle("B_s0_bar");
        }
    }
}

/// Built-in baryon and antibaryon particle definitions.
pub mod baryons {
    use super::*;

    const PROTON_RECORD: ParticleRecord = ParticleRecord::new(
        ParticleKind::Baryon(1),
        "p",
        SpeciesRecord::Pair("proton", "antiproton"),
        1,
        2212,
    )
    .parity(Parity::Positive)
    .charge(1)
    .isospin(1)
    .flavor(0, 0, 0)
    .mass(0.93827208816);
    const NEUTRON_RECORD: ParticleRecord = ParticleRecord::new(
        ParticleKind::Baryon(1),
        "n",
        SpeciesRecord::Pair("neutron", "antineutron"),
        1,
        2112,
    )
    .parity(Parity::Positive)
    .charge(0)
    .isospin(1)
    .flavor(0, 0, 0)
    .mass(0.93956542052);
    const LAMBDA_RECORD: ParticleRecord = ParticleRecord::new(
        ParticleKind::Baryon(1),
        "Lambda0",
        SpeciesRecord::Pair("Lambda0", "Lambda0_bar"),
        1,
        3122,
    )
    .parity(Parity::Positive)
    .charge(0)
    .isospin(0)
    .flavor(-1, 0, 0)
    .mass(1.115683);
    const SIGMA_PLUS_RECORD: ParticleRecord = ParticleRecord::new(
        ParticleKind::Baryon(1),
        "Sigma+",
        SpeciesRecord::Pair("Sigma+", "Sigma-_bar"),
        1,
        3222,
    )
    .parity(Parity::Positive)
    .charge(1)
    .isospin(2)
    .flavor(-1, 0, 0)
    .mass(1.18937);
    const SIGMA_ZERO_RECORD: ParticleRecord = ParticleRecord::new(
        ParticleKind::Baryon(1),
        "Sigma0",
        SpeciesRecord::Pair("Sigma0", "Sigma0_bar"),
        1,
        3212,
    )
    .parity(Parity::Positive)
    .charge(0)
    .isospin(2)
    .flavor(-1, 0, 0)
    .mass(1.192642);
    const SIGMA_MINUS_RECORD: ParticleRecord = ParticleRecord::new(
        ParticleKind::Baryon(1),
        "Sigma-",
        SpeciesRecord::Pair("Sigma-", "Sigma+_bar"),
        1,
        3112,
    )
    .parity(Parity::Positive)
    .charge(-1)
    .isospin(2)
    .flavor(-1, 0, 0)
    .mass(1.197449);
    const XI_ZERO_RECORD: ParticleRecord = ParticleRecord::new(
        ParticleKind::Baryon(1),
        "Xi0",
        SpeciesRecord::Pair("Xi0", "Xi0_bar"),
        1,
        3322,
    )
    .parity(Parity::Positive)
    .charge(0)
    .isospin(1)
    .flavor(-2, 0, 0)
    .mass(1.31486);
    const XI_MINUS_RECORD: ParticleRecord = ParticleRecord::new(
        ParticleKind::Baryon(1),
        "Xi-",
        SpeciesRecord::Pair("Xi-", "Xi+_bar"),
        1,
        3312,
    )
    .parity(Parity::Positive)
    .charge(-1)
    .isospin(1)
    .flavor(-2, 0, 0)
    .mass(1.32171);
    const OMEGA_MINUS_RECORD: ParticleRecord = ParticleRecord::new(
        ParticleKind::Baryon(1),
        "Omega-",
        SpeciesRecord::Pair("Omega-", "Omega+_bar"),
        3,
        3334,
    )
    .parity(Parity::Positive)
    .charge(-1)
    .isospin(0)
    .flavor(-3, 0, 0)
    .mass(1.67245);

    builtins! {
        /// Proton particle properties.
        PROTON => PROTON_RECORD;
        /// Antiproton particle properties.
        ANTIPROTON => PROTON_RECORD.antiparticle("p_bar");
        /// Neutron particle properties.
        NEUTRON => NEUTRON_RECORD;
        /// Antineutron particle properties.
        ANTINEUTRON => NEUTRON_RECORD.antiparticle("n_bar");
        /// Lambda-baryon particle properties.
        LAMBDA => LAMBDA_RECORD;
        /// Anti-Lambda-baryon particle properties.
        ANTILAMBDA => LAMBDA_RECORD.antiparticle("Lambda0_bar");
        /// Positively charged Sigma-baryon particle properties.
        SIGMA_PLUS => SIGMA_PLUS_RECORD;
        /// Negatively charged anti-Sigma-baryon particle properties.
        ANTISIGMA_MINUS => SIGMA_PLUS_RECORD.antiparticle("Sigma-_bar");
        /// Neutral Sigma-baryon particle properties.
        SIGMA_ZERO => SIGMA_ZERO_RECORD;
        /// Neutral anti-Sigma-baryon particle properties.
        ANTISIGMA_ZERO => SIGMA_ZERO_RECORD.antiparticle("Sigma0_bar");
        /// Negatively charged Sigma-baryon particle properties.
        SIGMA_MINUS => SIGMA_MINUS_RECORD;
        /// Positively charged anti-Sigma-baryon particle properties.
        ANTISIGMA_PLUS => SIGMA_MINUS_RECORD.antiparticle("Sigma+_bar");
        /// Neutral Xi-baryon particle properties.
        XI_ZERO => XI_ZERO_RECORD;
        /// Neutral anti-Xi-baryon particle properties.
        ANTI_XI_ZERO => XI_ZERO_RECORD.antiparticle("Xi0_bar");
        /// Negatively charged Xi-baryon particle properties.
        XI_MINUS => XI_MINUS_RECORD;
        /// Positively charged anti-Xi-baryon particle properties.
        ANTI_XI_PLUS => XI_MINUS_RECORD.antiparticle("Xi+_bar");
        /// Negatively charged Omega-baryon particle properties.
        OMEGA_MINUS => OMEGA_MINUS_RECORD;
        /// Positively charged anti-Omega-baryon particle properties.
        ANTI_OMEGA_PLUS => OMEGA_MINUS_RECORD.antiparticle("Omega+_bar");
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn catalog() -> [(&'static str, &'static ParticleProperties); 64] {
        [
            ("PHOTON", &PHOTON),
            ("W_PLUS", &W_PLUS),
            ("W_MINUS", &W_MINUS),
            ("Z_BOSON", &Z_BOSON),
            ("HIGGS", &HIGGS),
            ("ELECTRON", &ELECTRON),
            ("POSITRON", &POSITRON),
            ("MUON", &MUON),
            ("ANTIMUON", &ANTIMUON),
            ("TAU", &TAU),
            ("ANTITAU", &ANTITAU),
            ("ELECTRON_NEUTRINO", &ELECTRON_NEUTRINO),
            ("ELECTRON_ANTINEUTRINO", &ELECTRON_ANTINEUTRINO),
            ("MUON_NEUTRINO", &MUON_NEUTRINO),
            ("MUON_ANTINEUTRINO", &MUON_ANTINEUTRINO),
            ("TAU_NEUTRINO", &TAU_NEUTRINO),
            ("TAU_ANTINEUTRINO", &TAU_ANTINEUTRINO),
            ("PI_PLUS", &PI_PLUS),
            ("PI_MINUS", &PI_MINUS),
            ("PI_ZERO", &PI_ZERO),
            ("K_PLUS", &K_PLUS),
            ("K_MINUS", &K_MINUS),
            ("K_ZERO", &K_ZERO),
            ("K_ZERO_BAR", &K_ZERO_BAR),
            ("K_SHORT", &K_SHORT),
            ("K_LONG", &K_LONG),
            ("ETA", &ETA),
            ("ETA_PRIME", &ETA_PRIME),
            ("RHO_PLUS", &RHO_PLUS),
            ("RHO_MINUS", &RHO_MINUS),
            ("RHO_ZERO", &RHO_ZERO),
            ("OMEGA", &OMEGA),
            ("PHI", &PHI),
            ("J_PSI", &J_PSI),
            ("D_PLUS", &D_PLUS),
            ("D_MINUS", &D_MINUS),
            ("D_ZERO", &D_ZERO),
            ("D_ZERO_BAR", &D_ZERO_BAR),
            ("D_S_PLUS", &D_S_PLUS),
            ("D_S_MINUS", &D_S_MINUS),
            ("B_PLUS", &B_PLUS),
            ("B_MINUS", &B_MINUS),
            ("B_ZERO", &B_ZERO),
            ("B_ZERO_BAR", &B_ZERO_BAR),
            ("B_S_ZERO", &B_S_ZERO),
            ("B_S_ZERO_BAR", &B_S_ZERO_BAR),
            ("PROTON", &PROTON),
            ("ANTIPROTON", &ANTIPROTON),
            ("NEUTRON", &NEUTRON),
            ("ANTINEUTRON", &ANTINEUTRON),
            ("LAMBDA", &LAMBDA),
            ("ANTILAMBDA", &ANTILAMBDA),
            ("SIGMA_PLUS", &SIGMA_PLUS),
            ("ANTISIGMA_MINUS", &ANTISIGMA_MINUS),
            ("SIGMA_ZERO", &SIGMA_ZERO),
            ("ANTISIGMA_ZERO", &ANTISIGMA_ZERO),
            ("SIGMA_MINUS", &SIGMA_MINUS),
            ("ANTISIGMA_PLUS", &ANTISIGMA_PLUS),
            ("XI_ZERO", &XI_ZERO),
            ("ANTI_XI_ZERO", &ANTI_XI_ZERO),
            ("XI_MINUS", &XI_MINUS),
            ("ANTI_XI_PLUS", &ANTI_XI_PLUS),
            ("OMEGA_MINUS", &OMEGA_MINUS),
            ("ANTI_OMEGA_PLUS", &ANTI_OMEGA_PLUS),
        ]
    }

    fn pairs() -> [(&'static ParticleProperties, &'static ParticleProperties); 26] {
        [
            (&W_PLUS, &W_MINUS),
            (&ELECTRON, &POSITRON),
            (&MUON, &ANTIMUON),
            (&TAU, &ANTITAU),
            (&ELECTRON_NEUTRINO, &ELECTRON_ANTINEUTRINO),
            (&MUON_NEUTRINO, &MUON_ANTINEUTRINO),
            (&TAU_NEUTRINO, &TAU_ANTINEUTRINO),
            (&PI_PLUS, &PI_MINUS),
            (&K_PLUS, &K_MINUS),
            (&K_ZERO, &K_ZERO_BAR),
            (&RHO_PLUS, &RHO_MINUS),
            (&D_PLUS, &D_MINUS),
            (&D_ZERO, &D_ZERO_BAR),
            (&D_S_PLUS, &D_S_MINUS),
            (&B_PLUS, &B_MINUS),
            (&B_ZERO, &B_ZERO_BAR),
            (&B_S_ZERO, &B_S_ZERO_BAR),
            (&PROTON, &ANTIPROTON),
            (&NEUTRON, &ANTINEUTRON),
            (&LAMBDA, &ANTILAMBDA),
            (&SIGMA_PLUS, &ANTISIGMA_MINUS),
            (&SIGMA_ZERO, &ANTISIGMA_ZERO),
            (&SIGMA_MINUS, &ANTISIGMA_PLUS),
            (&XI_ZERO, &ANTI_XI_ZERO),
            (&XI_MINUS, &ANTI_XI_PLUS),
            (&OMEGA_MINUS, &ANTI_OMEGA_PLUS),
        ]
    }

    #[test]
    fn catalog_matches_compatibility_fixture() {
        let actual = serde_json::to_value(&catalog()[..]).unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/builtin_particles.json")).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn catalog_records_are_valid_and_have_unique_pdg_ids() {
        let mut ids = HashSet::new();
        for (name, particle) in catalog() {
            particle
                .check_invariants()
                .unwrap_or_else(|error| panic!("{name} violates particle invariants: {error}"));
            if let Some(mass) = particle.mass {
                assert!(mass.is_finite() && mass >= 0.0, "invalid mass for {name}");
            }
            let pdg_id = particle.id("pdg").and_then(|id| id.code_value()).unwrap();
            assert!(ids.insert(pdg_id), "duplicate PDG ID {pdg_id} for {name}");
        }
    }

    #[test]
    fn antiparticle_records_have_reciprocal_species_and_signed_quantum_numbers() {
        for (particle, antiparticle) in pairs() {
            assert!(particle.is_antiparticle_of(antiparticle));
            assert!(antiparticle.is_antiparticle_of(particle));
            assert_eq!(particle.spin, antiparticle.spin);
            assert_eq!(particle.parity, antiparticle.parity);
            assert_eq!(particle.isospin, antiparticle.isospin);
            assert_eq!(particle.mass, antiparticle.mass);
            assert_eq!(particle.charge.map(|value| -value), antiparticle.charge);
            assert_eq!(
                particle.strangeness.map(|value| -value),
                antiparticle.strangeness
            );
            assert_eq!(particle.charm.map(|value| -value), antiparticle.charm);
            assert_eq!(
                particle.bottomness.map(|value| -value),
                antiparticle.bottomness
            );
            assert_eq!(particle.topness.map(|value| -value), antiparticle.topness);
            assert_eq!(
                particle.baryon_number.map(|value| -value),
                antiparticle.baryon_number
            );
            assert_eq!(
                particle.electron_lepton_number.map(|value| -value),
                antiparticle.electron_lepton_number
            );
            assert_eq!(
                particle.muon_lepton_number.map(|value| -value),
                antiparticle.muon_lepton_number
            );
            assert_eq!(
                particle.tau_lepton_number.map(|value| -value),
                antiparticle.tau_lepton_number
            );
            assert_eq!(
                particle
                    .id("pdg")
                    .and_then(|id| id.code_value())
                    .map(|id| -id),
                antiparticle.id("pdg").and_then(|id| id.code_value())
            );
        }
    }
}
