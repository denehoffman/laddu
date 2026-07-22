use std::collections::HashMap;

use laddu_physics::quantum::{ExternalId, ParticleProperties};
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
    #[new]
    #[pyo3(signature = (
        name=None,
        *,
        species=None,
        antiparticle_species=None,
        self_conjugate=None,
        spin: "S | J | L | int | float | fractions.Fraction | None"=None,
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
        let mut particle = ParticleProperties::unknown();
        if let Some(name) = name {
            particle = particle.with_name(name);
        }
        match (species, antiparticle_species) {
            (Some(species), Some(antiparticle)) => {
                particle = particle
                    .with_species_names(species, antiparticle)
                    .map_err(to_py_err)?;
            }
            (Some(species), None) => {
                particle = particle.with_species(species).map_err(to_py_err)?;
            }
            (None, Some(antiparticle)) => {
                particle = particle
                    .with_antiparticle_species(antiparticle)
                    .map_err(to_py_err)?;
            }
            (None, None) => {}
        }
        if let Some(value) = self_conjugate {
            particle = particle.with_self_conjugate(value).map_err(to_py_err)?;
        }
        if let Some(spin) = spin {
            particle = particle.with_spin(extract_spin(spin)?);
        }
        if let Some(value) = parity {
            particle = particle.with_parity(extract_parity(value)?);
        }
        if let Some(value) = c_parity {
            particle = particle
                .with_c_parity(extract_parity(value)?)
                .map_err(to_py_err)?;
        }
        if let Some(value) = g_parity {
            particle = particle.with_g_parity(extract_parity(value)?);
        }
        if let Some(value) = charge {
            particle = particle.with_charge(value);
        }
        if let Some(value) = isospin {
            particle = particle.with_isospin(value.inner.clone());
        }
        if let Some(value) = strangeness {
            particle = particle.with_strangeness(value).map_err(to_py_err)?;
        }
        if let Some(value) = charm {
            particle = particle.with_charm(value).map_err(to_py_err)?;
        }
        if let Some(value) = bottomness {
            particle = particle.with_bottomness(value).map_err(to_py_err)?;
        }
        if let Some(value) = topness {
            particle = particle.with_topness(value).map_err(to_py_err)?;
        }
        if let Some(value) = baryon_number {
            particle = particle.with_baryon_number(value).map_err(to_py_err)?;
        }
        if let Some(value) = electron_lepton_number {
            particle = particle
                .with_electron_lepton_number(value)
                .map_err(to_py_err)?;
        }
        if let Some(value) = muon_lepton_number {
            particle = particle.with_muon_lepton_number(value).map_err(to_py_err)?;
        }
        if let Some(value) = tau_lepton_number {
            particle = particle.with_tau_lepton_number(value).map_err(to_py_err)?;
        }
        if let Some(value) = statistics {
            particle.statistics = Some(value.inner);
        }
        if let Some(value) = mass {
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(
                    "particle mass must be finite and non-negative",
                ));
            }
            particle = particle.with_mass(value);
        }
        if let Some(ids) = ids {
            particle = particle.with_ids(
                ids.into_iter()
                    .map(|(namespace, value)| (namespace, ExternalId::code(value))),
            );
        }
        Ok(particle.into())
    }

    #[staticmethod]
    fn unknown() -> Self {
        ParticleProperties::unknown().into()
    }

    #[staticmethod]
    fn jp(j: &Bound<'_, PyAny>, parity: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(ParticleProperties::jp(extract_spin(j)?, extract_parity(parity)?).into())
    }

    #[staticmethod]
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
    fn boson(j: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(ParticleProperties::boson(extract_l(j)?).into())
    }

    #[staticmethod]
    #[pyo3(signature = (electron=0, muon=0, tau=0))]
    fn lepton(electron: i32, muon: i32, tau: i32) -> Self {
        ParticleProperties::lepton(electron, muon, tau).into()
    }

    #[staticmethod]
    fn hadron() -> Self {
        ParticleProperties::hadron().into()
    }

    #[staticmethod]
    fn meson() -> Self {
        ParticleProperties::meson().into()
    }

    #[staticmethod]
    #[pyo3(signature = (baryon_number=1))]
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
    fn name(&self) -> PyResult<String> {
        required(&self.inner.name, "name")
    }
    #[getter]
    fn name_checked(&self) -> Option<String> {
        self.inner.name.clone()
    }
    #[getter]
    fn species(&self) -> PyResult<String> {
        required(&self.inner.species, "species")
    }
    #[getter]
    fn species_checked(&self) -> Option<String> {
        self.inner.species.clone()
    }
    #[getter]
    fn antiparticle_species(&self) -> PyResult<String> {
        required(&self.inner.antiparticle_species, "antiparticle_species")
    }
    #[getter]
    fn antiparticle_species_checked(&self) -> Option<String> {
        self.inner.antiparticle_species.clone()
    }
    #[getter]
    fn self_conjugate(&self) -> PyResult<bool> {
        required(&self.inner.self_conjugate, "self_conjugate")
    }
    #[getter]
    fn self_conjugate_checked(&self) -> Option<bool> {
        self.inner.self_conjugate
    }
    #[getter]
    fn spin(&self) -> PyResult<PyS> {
        Ok(PyS {
            inner: required(&self.inner.spin, "spin")?,
        })
    }
    #[getter]
    fn spin_checked(&self) -> Option<PyS> {
        self.inner.spin.map(|inner| PyS { inner })
    }
    #[getter]
    fn parity(&self) -> PyResult<PyParity> {
        Ok(required(&self.inner.parity, "parity")?.into())
    }
    #[getter]
    fn parity_checked(&self) -> Option<PyParity> {
        self.inner.parity.map(PyParity::from)
    }
    #[getter]
    fn c_parity(&self) -> PyResult<PyParity> {
        Ok(required(&self.inner.c_parity, "c_parity")?.into())
    }
    #[getter]
    fn c_parity_checked(&self) -> Option<PyParity> {
        self.inner.c_parity.map(PyParity::from)
    }
    #[getter]
    fn g_parity(&self) -> PyResult<PyParity> {
        Ok(required(&self.inner.g_parity, "g_parity")?.into())
    }
    #[getter]
    fn g_parity_checked(&self) -> Option<PyParity> {
        self.inner.g_parity.map(PyParity::from)
    }
    #[getter]
    fn charge(&self) -> PyResult<i32> {
        required(&self.inner.charge, "charge")
    }
    #[getter]
    fn charge_checked(&self) -> Option<i32> {
        self.inner.charge
    }
    #[getter]
    fn isospin(&self) -> PyResult<PyIsospin> {
        Ok(PyIsospin {
            inner: required(&self.inner.isospin, "isospin")?,
        })
    }
    #[getter]
    fn isospin_checked(&self) -> Option<PyIsospin> {
        self.inner.isospin.clone().map(|inner| PyIsospin { inner })
    }
    #[getter]
    fn strangeness(&self) -> PyResult<i32> {
        required(&self.inner.strangeness, "strangeness")
    }
    #[getter]
    fn strangeness_checked(&self) -> Option<i32> {
        self.inner.strangeness
    }
    #[getter]
    fn charm(&self) -> PyResult<i32> {
        required(&self.inner.charm, "charm")
    }
    #[getter]
    fn charm_checked(&self) -> Option<i32> {
        self.inner.charm
    }
    #[getter]
    fn bottomness(&self) -> PyResult<i32> {
        required(&self.inner.bottomness, "bottomness")
    }
    #[getter]
    fn bottomness_checked(&self) -> Option<i32> {
        self.inner.bottomness
    }
    #[getter]
    fn topness(&self) -> PyResult<i32> {
        required(&self.inner.topness, "topness")
    }
    #[getter]
    fn topness_checked(&self) -> Option<i32> {
        self.inner.topness
    }
    #[getter]
    fn baryon_number(&self) -> PyResult<i32> {
        required(&self.inner.baryon_number, "baryon_number")
    }
    #[getter]
    fn baryon_number_checked(&self) -> Option<i32> {
        self.inner.baryon_number
    }
    #[getter]
    fn electron_lepton_number(&self) -> PyResult<i32> {
        required(&self.inner.electron_lepton_number, "electron_lepton_number")
    }
    #[getter]
    fn electron_lepton_number_checked(&self) -> Option<i32> {
        self.inner.electron_lepton_number
    }
    #[getter]
    fn muon_lepton_number(&self) -> PyResult<i32> {
        required(&self.inner.muon_lepton_number, "muon_lepton_number")
    }
    #[getter]
    fn muon_lepton_number_checked(&self) -> Option<i32> {
        self.inner.muon_lepton_number
    }
    #[getter]
    fn tau_lepton_number(&self) -> PyResult<i32> {
        required(&self.inner.tau_lepton_number, "tau_lepton_number")
    }
    #[getter]
    fn tau_lepton_number_checked(&self) -> Option<i32> {
        self.inner.tau_lepton_number
    }
    #[getter]
    fn statistics(&self) -> PyResult<PyStatistics> {
        Ok(required(&self.inner.statistics, "statistics")?.into())
    }
    #[getter]
    fn statistics_checked(&self) -> Option<PyStatistics> {
        self.inner.statistics.map(PyStatistics::from)
    }
    #[getter]
    fn mass(&self) -> PyResult<f64> {
        required(&self.inner.mass, "mass")
    }
    #[getter]
    fn mass_checked(&self) -> Option<f64> {
        self.inner.mass
    }

    #[getter]
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
