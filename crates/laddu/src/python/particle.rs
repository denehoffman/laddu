use std::collections::HashMap;

use laddu_physics::quantum::{ExternalId, ParticleProperties, ParticlePropertiesPatch};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyModule},
};

use super::{
    error::to_py_err,
    quantum::{PyIsospin, PyParity, PyS, PyStatistics, extract_l, extract_parity, extract_spin},
};

fn required<T: Clone>(value: &Option<T>, property: &str) -> PyResult<T> {
    value
        .clone()
        .ok_or_else(|| PyValueError::new_err(format!("particle does not define `{property}`")))
}

#[pyclass(name = "Particle", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Physical and quantum-number metadata for a particle species.
///
/// Every property is optional. The ordinary property accessors raise
/// ``ValueError`` when metadata are absent; accessors ending in ``_checked``
/// return ``None`` instead.
///
/// Parameters
/// ----------
/// name : str, optional
///     Display name for this particle state.
/// species, antiparticle_species : str, optional
///     Particle and antiparticle species names.
/// self_conjugate : bool, optional
///     Whether particle and antiparticle are identical.
/// spin : S, J, L, int, float, or fractions.Fraction, optional
///     Spin quantum number.
/// parity, c_parity, g_parity : Parity, int, or str, optional
///     Intrinsic, charge-conjugation, and G-parity eigenvalues.
/// charge : int, optional
///     Electric charge in units of the positron charge.
/// isospin : Isospin, optional
///     Total isospin and third component.
/// strangeness, charm, bottomness, topness : int, optional
///     Flavor quantum numbers.
/// baryon_number : int, optional
///     Baryon number.
/// electron_lepton_number, muon_lepton_number, tau_lepton_number : int, optional
///     Flavor lepton numbers.
/// statistics : Statistics, optional
///     Bosonic or fermionic exchange statistics.
/// mass : float, optional
///     Non-negative mass in the units used by event four-vectors.
/// ids : dict[str, int], optional
///     External identifiers keyed by namespace, such as ``{"pdg": 211}``.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> pion = ld.Particle("pi+", spin=0, parity="-", charge=1, mass=0.13957)
/// >>> pion.charge
/// 1
/// >>> pion.c_parity_checked is None
/// True
pub struct PyParticle {
    pub(crate) inner: ParticleProperties,
}

impl From<ParticleProperties> for PyParticle {
    fn from(inner: ParticleProperties) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyParticle {
    /// Construct particle metadata.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a quantum number, species relation, or mass is invalid.
    #[new]
    #[pyo3(signature = (
        name=None,
        *,
        species=None,
        antiparticle_species=None,
        self_conjugate=None,
        spin: "S | J | L | int | float | None"=None,
        parity: "Parity | int | str | None"=None,
        c_parity: "Parity | int | str | None"=None,
        g_parity: "Parity | int | str | None"=None,
        charge=None,
        isospin: "Isospin | None"=None,
        strangeness=None,
        charm=None,
        bottomness=None,
        topness=None,
        baryon_number=None,
        electron_lepton_number=None,
        muon_lepton_number=None,
        tau_lepton_number=None,
        statistics: "Statistics | None"=None,
        mass=None,
        ids=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: Option<String>,
        species: Option<String>,
        antiparticle_species: Option<String>,
        self_conjugate: Option<bool>,
        spin: Option<&Bound<'_, PyAny>>,
        parity: Option<&Bound<'_, PyAny>>,
        c_parity: Option<&Bound<'_, PyAny>>,
        g_parity: Option<&Bound<'_, PyAny>>,
        charge: Option<i32>,
        isospin: Option<PyRef<'_, PyIsospin>>,
        strangeness: Option<i32>,
        charm: Option<i32>,
        bottomness: Option<i32>,
        topness: Option<i32>,
        baryon_number: Option<i32>,
        electron_lepton_number: Option<i32>,
        muon_lepton_number: Option<i32>,
        tau_lepton_number: Option<i32>,
        statistics: Option<PyRef<'_, PyStatistics>>,
        mass: Option<f64>,
        ids: Option<HashMap<String, i64>>,
    ) -> PyResult<Self> {
        if let Some(value) = mass
            && (!value.is_finite() || value < 0.0)
        {
            return Err(PyValueError::new_err(
                "particle mass must be finite and non-negative",
            ));
        }

        let patch = ParticlePropertiesPatch {
            name,
            species,
            antiparticle_species,
            self_conjugate,
            spin: spin.map(extract_spin).transpose()?,
            parity: parity.map(extract_parity).transpose()?,
            c_parity: c_parity.map(extract_parity).transpose()?,
            g_parity: g_parity.map(extract_parity).transpose()?,
            charge,
            isospin: isospin.map(|value| value.inner),
            strangeness,
            charm,
            bottomness,
            topness,
            baryon_number,
            electron_lepton_number,
            muon_lepton_number,
            tau_lepton_number,
            statistics: statistics.map(|value| value.inner),
            mass,
            ids: ids.map(|ids| {
                ids.into_iter()
                    .map(|(namespace, value)| (namespace, ExternalId::code(value)))
                    .collect()
            }),
        };
        Ok(ParticleProperties::unknown()
            .apply_patch(patch)
            .map_err(to_py_err)?
            .into())
    }

    #[staticmethod]
    /// Create a particle with no known properties.
    fn unknown() -> Self {
        ParticleProperties::unknown().into()
    }

    #[staticmethod]
    /// Create a particle from spin and intrinsic parity.
    ///
    /// Parameters
    /// ----------
    /// j : S, J, L, int, float, or fractions.Fraction
    ///     Spin quantum number.
    /// parity : Parity, int, or str
    ///     Intrinsic parity.
    #[pyo3(signature = (
        j: "J | S | L | int | float",
        *,
        parity: "Parity | int | str"
    ))]
    fn jp(j: &Bound<'_, PyAny>, parity: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(ParticleProperties::jp(extract_spin(j)?, extract_parity(parity)?).into())
    }

    #[staticmethod]
    #[pyo3(signature = (
        j: "J | S | L | int | float",
        *,
        parity: "Parity | int | str",
        c_parity: "Parity | int | str"
    ))]
    /// Create a particle from spin, parity, and C-parity.
    fn jpc(
        j: &Bound<'_, PyAny>,
        parity: &Bound<'_, PyAny>,
        c_parity: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(ParticleProperties::jpc(
            extract_spin(j)?,
            extract_parity(parity)?,
            extract_parity(c_parity)?,
        )
        .into())
    }

    #[staticmethod]
    #[pyo3(signature = (j: "L | J | S | int | float"))]
    /// Create a boson with integer spin.
    fn boson(j: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(ParticleProperties::boson(extract_l(j)?).into())
    }

    #[staticmethod]
    #[pyo3(signature = (*, electron=0, muon=0, tau=0))]
    /// Create a fermion with specified flavor lepton numbers.
    fn lepton(electron: i32, muon: i32, tau: i32) -> Self {
        ParticleProperties::lepton(electron, muon, tau).into()
    }

    #[staticmethod]
    /// Create generic hadron metadata.
    fn hadron() -> Self {
        ParticleProperties::hadron().into()
    }

    #[staticmethod]
    /// Create generic meson metadata.
    fn meson() -> Self {
        ParticleProperties::meson().into()
    }

    #[staticmethod]
    #[pyo3(signature = (baryon_number=1))]
    /// Create generic baryon metadata.
    fn baryon(baryon_number: i32) -> Self {
        ParticleProperties::baryon(baryon_number).into()
    }

    fn __repr__(&self) -> String {
        match &self.inner.name {
            Some(name) => format!("Particle({name:?})"),
            None => "Particle()".to_owned(),
        }
    }

    #[getter]
    /// str: Particle display name; raises if unknown.
    fn name(&self) -> PyResult<String> {
        required(&self.inner.name, "name")
    }
    #[getter]
    /// str or None: Particle display name.
    fn name_checked(&self) -> Option<String> {
        self.inner.name.clone()
    }
    #[getter]
    /// str: Particle species name; raises if unknown.
    fn species(&self) -> PyResult<String> {
        required(&self.inner.species, "species")
    }
    #[getter]
    /// str or None: Particle species name.
    fn species_checked(&self) -> Option<String> {
        self.inner.species.clone()
    }
    #[getter]
    /// str: Antiparticle species name; raises if unknown.
    fn antiparticle_species(&self) -> PyResult<String> {
        required(&self.inner.antiparticle_species, "antiparticle_species")
    }
    #[getter]
    /// str or None: Antiparticle species name.
    fn antiparticle_species_checked(&self) -> Option<String> {
        self.inner.antiparticle_species.clone()
    }
    #[getter]
    /// bool: Whether the particle is self-conjugate; raises if unknown.
    fn self_conjugate(&self) -> PyResult<bool> {
        required(&self.inner.self_conjugate, "self_conjugate")
    }
    #[getter]
    /// bool or None: Whether the particle is self-conjugate.
    fn self_conjugate_checked(&self) -> Option<bool> {
        self.inner.self_conjugate
    }
    #[getter]
    /// S: Spin quantum number; raises if unknown.
    fn spin(&self) -> PyResult<PyS> {
        Ok(PyS {
            inner: required(&self.inner.spin, "spin")?,
        })
    }
    #[getter]
    /// S or None: Spin quantum number.
    fn spin_checked(&self) -> Option<PyS> {
        self.inner.spin.map(|inner| PyS { inner })
    }
    #[getter]
    /// Parity: Intrinsic parity; raises if unknown.
    fn parity(&self) -> PyResult<PyParity> {
        Ok(required(&self.inner.parity, "parity")?.into())
    }
    #[getter]
    /// Parity or None: Intrinsic parity.
    fn parity_checked(&self) -> Option<PyParity> {
        self.inner.parity.map(PyParity::from)
    }
    #[getter]
    /// Parity: Charge-conjugation parity; raises if unknown.
    fn c_parity(&self) -> PyResult<PyParity> {
        Ok(required(&self.inner.c_parity, "c_parity")?.into())
    }
    #[getter]
    /// Parity or None: Charge-conjugation parity.
    fn c_parity_checked(&self) -> Option<PyParity> {
        self.inner.c_parity.map(PyParity::from)
    }
    #[getter]
    /// Parity: G-parity; raises if unknown.
    fn g_parity(&self) -> PyResult<PyParity> {
        Ok(required(&self.inner.g_parity, "g_parity")?.into())
    }
    #[getter]
    /// Parity or None: G-parity.
    fn g_parity_checked(&self) -> Option<PyParity> {
        self.inner.g_parity.map(PyParity::from)
    }
    #[getter]
    /// int: Electric charge; raises if unknown.
    fn charge(&self) -> PyResult<i32> {
        required(&self.inner.charge, "charge")
    }
    #[getter]
    /// int or None: Electric charge.
    fn charge_checked(&self) -> Option<i32> {
        self.inner.charge
    }
    #[getter]
    /// Isospin: Isospin state; raises if unknown.
    fn isospin(&self) -> PyResult<PyIsospin> {
        Ok(PyIsospin {
            inner: required(&self.inner.isospin, "isospin")?,
        })
    }
    #[getter]
    /// Isospin or None: Isospin state.
    fn isospin_checked(&self) -> Option<PyIsospin> {
        self.inner.isospin.map(|inner| PyIsospin { inner })
    }
    #[getter]
    /// int: Strangeness; raises if unknown.
    fn strangeness(&self) -> PyResult<i32> {
        required(&self.inner.strangeness, "strangeness")
    }
    #[getter]
    /// int or None: Strangeness.
    fn strangeness_checked(&self) -> Option<i32> {
        self.inner.strangeness
    }
    #[getter]
    /// int: Charm; raises if unknown.
    fn charm(&self) -> PyResult<i32> {
        required(&self.inner.charm, "charm")
    }
    #[getter]
    /// int or None: Charm.
    fn charm_checked(&self) -> Option<i32> {
        self.inner.charm
    }
    #[getter]
    /// int: Bottomness; raises if unknown.
    fn bottomness(&self) -> PyResult<i32> {
        required(&self.inner.bottomness, "bottomness")
    }
    #[getter]
    /// int or None: Bottomness.
    fn bottomness_checked(&self) -> Option<i32> {
        self.inner.bottomness
    }
    #[getter]
    /// int: Topness; raises if unknown.
    fn topness(&self) -> PyResult<i32> {
        required(&self.inner.topness, "topness")
    }
    #[getter]
    /// int or None: Topness.
    fn topness_checked(&self) -> Option<i32> {
        self.inner.topness
    }
    #[getter]
    /// int: Baryon number; raises if unknown.
    fn baryon_number(&self) -> PyResult<i32> {
        required(&self.inner.baryon_number, "baryon_number")
    }
    #[getter]
    /// int or None: Baryon number.
    fn baryon_number_checked(&self) -> Option<i32> {
        self.inner.baryon_number
    }
    #[getter]
    /// int: Electron lepton number; raises if unknown.
    fn electron_lepton_number(&self) -> PyResult<i32> {
        required(&self.inner.electron_lepton_number, "electron_lepton_number")
    }
    #[getter]
    /// int or None: Electron lepton number.
    fn electron_lepton_number_checked(&self) -> Option<i32> {
        self.inner.electron_lepton_number
    }
    #[getter]
    /// int: Muon lepton number; raises if unknown.
    fn muon_lepton_number(&self) -> PyResult<i32> {
        required(&self.inner.muon_lepton_number, "muon_lepton_number")
    }
    #[getter]
    /// int or None: Muon lepton number.
    fn muon_lepton_number_checked(&self) -> Option<i32> {
        self.inner.muon_lepton_number
    }
    #[getter]
    /// int: Tau lepton number; raises if unknown.
    fn tau_lepton_number(&self) -> PyResult<i32> {
        required(&self.inner.tau_lepton_number, "tau_lepton_number")
    }
    #[getter]
    /// int or None: Tau lepton number.
    fn tau_lepton_number_checked(&self) -> Option<i32> {
        self.inner.tau_lepton_number
    }
    #[getter]
    /// Statistics: Exchange statistics; raises if unknown.
    fn statistics(&self) -> PyResult<PyStatistics> {
        Ok(required(&self.inner.statistics, "statistics")?.into())
    }
    #[getter]
    /// Statistics or None: Exchange statistics.
    fn statistics_checked(&self) -> Option<PyStatistics> {
        self.inner.statistics.map(PyStatistics::from)
    }
    #[getter]
    /// float: Particle mass; raises if unknown.
    fn mass(&self) -> PyResult<f64> {
        required(&self.inner.mass, "mass")
    }
    #[getter]
    /// float or None: Particle mass.
    fn mass_checked(&self) -> Option<f64> {
        self.inner.mass
    }

    #[getter]
    /// dict[str, int]: External numeric identifiers by namespace.
    fn ids(&self) -> HashMap<String, i64> {
        self.inner
            .ids
            .iter()
            .filter_map(|(namespace, value)| {
                value.code_value().map(|value| (namespace.clone(), value))
            })
            .collect()
    }
}

#[pymodule(submodule)]
/// Built-in particle definitions, including Standard Model hadrons and leptons.
pub mod particles {
    use super::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        macro_rules! add_particles {
            ($($name:ident),+ $(,)?) => {$({
                module.add(
                    stringify!($name),
                    Py::new(module.py(), PyParticle::from((*laddu_physics::quantum::builtin::$name).clone()))?,
                )?;
            })+};
        }
        add_particles!(
            PHOTON,
            W_PLUS,
            W_MINUS,
            Z_BOSON,
            HIGGS,
            ELECTRON,
            POSITRON,
            MUON,
            ANTIMUON,
            TAU,
            ANTITAU,
            ELECTRON_NEUTRINO,
            ELECTRON_ANTINEUTRINO,
            MUON_NEUTRINO,
            MUON_ANTINEUTRINO,
            TAU_NEUTRINO,
            TAU_ANTINEUTRINO,
            PI_PLUS,
            PI_MINUS,
            PI_ZERO,
            K_PLUS,
            K_MINUS,
            K_ZERO,
            K_ZERO_BAR,
            K_SHORT,
            K_LONG,
            ETA,
            ETA_PRIME,
            RHO_PLUS,
            RHO_MINUS,
            RHO_ZERO,
            OMEGA,
            PHI,
            J_PSI,
            D_PLUS,
            D_MINUS,
            D_ZERO,
            D_ZERO_BAR,
            D_S_PLUS,
            D_S_MINUS,
            B_PLUS,
            B_MINUS,
            B_ZERO,
            B_ZERO_BAR,
            B_S_ZERO,
            B_S_ZERO_BAR,
            PROTON,
            ANTIPROTON,
            NEUTRON,
            ANTINEUTRON,
            LAMBDA,
            ANTILAMBDA,
            SIGMA_PLUS,
            ANTISIGMA_MINUS,
            SIGMA_ZERO,
            ANTISIGMA_ZERO,
            SIGMA_MINUS,
            ANTISIGMA_PLUS,
            XI_ZERO,
            ANTI_XI_ZERO,
            XI_MINUS,
            ANTI_XI_PLUS,
            OMEGA_MINUS,
            ANTI_OMEGA_PLUS,
        );
        Ok(())
    }
}

impl_json_methods!(PyParticle);
