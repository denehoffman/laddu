pub mod angular_momentum {
    use laddu_core::{
        allowed_projections, AngularMomentum, LadduError, LadduResult, OrbitalAngularMomentum,
        Projection,
    };
    use num::rational::Ratio;
    use pyo3::{
        prelude::*,
        types::{PyAny, PyBool, PyModule},
        IntoPyObjectExt,
    };
    type PyQuantumNumber = Py<PyAny>;

    pub fn parse_angular_momentum(input: &Bound<'_, PyAny>) -> PyResult<AngularMomentum> {
        Ok(parse_ratio_like(input).and_then(AngularMomentum::try_from)?)
    }

    fn parse_ratio_like(input: &Bound<'_, PyAny>) -> LadduResult<Ratio<i32>> {
        if input.is_instance_of::<PyBool>() {
            return Err(LadduError::Custom(
                "quantum number cannot be a bool".to_string(),
            ));
        }
        if let Ok(value) = input.extract::<i32>() {
            return Ok(Ratio::from_integer(value));
        }
        if let Ok(value) = input.extract::<f64>() {
            let twice = Projection::try_from(value)?.value();
            return Ok(Ratio::new(twice, 2));
        }
        let numerator = input
            .getattr("numerator")
            .and_then(|value| value.extract::<i32>());
        let denominator = input
            .getattr("denominator")
            .and_then(|value| value.extract::<i32>());
        if let (Ok(numerator), Ok(denominator)) = (numerator, denominator) {
            if denominator == 0 {
                return Err(LadduError::Custom(
                    "quantum number denominator cannot be zero".to_string(),
                ));
            }
            return Ok(Ratio::new(numerator, denominator));
        }
        Err(LadduError::Custom(
            "quantum number must be an int, float, or fractions.Fraction".to_string(),
        ))
    }

    pub fn parse_projection(input: &Bound<'_, PyAny>) -> PyResult<Projection> {
        Ok(parse_ratio_like(input).and_then(Projection::try_from)?)
    }

    pub fn parse_orbital_angular_momentum(
        input: &Bound<'_, PyAny>,
    ) -> PyResult<OrbitalAngularMomentum> {
        Ok(parse_ratio_like(input).and_then(OrbitalAngularMomentum::try_from)?)
    }

    pub fn angular_momentum_to_python(
        py: Python<'_>,
        angular_momentum: laddu_core::AngularMomentum,
    ) -> PyResult<PyQuantumNumber> {
        let twice = angular_momentum.value() as i32;
        if twice % 2 == 0 {
            Ok((twice / 2).into_bound_py_any(py)?.unbind())
        } else {
            let fractions = PyModule::import(py, "fractions")?;
            let fraction = fractions.getattr("Fraction")?;
            Ok(fraction.call1((twice, 2))?.unbind())
        }
    }

    pub fn projection_to_python(
        py: Python<'_>,
        projection: Projection,
    ) -> PyResult<PyQuantumNumber> {
        let twice = projection.value();
        if twice % 2 == 0 {
            Ok((twice / 2).into_bound_py_any(py)?.unbind())
        } else {
            let fractions = PyModule::import(py, "fractions")?;
            let fraction = fractions.getattr("Fraction")?;
            Ok(fraction.call1((twice, 2))?.unbind())
        }
    }

    /// Enumerate allowed spin projections.
    #[pyfunction(name = "allowed_projections")]
    pub fn py_allowed_projections(
        py: Python<'_>,
        spin: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyQuantumNumber>> {
        allowed_projections(parse_angular_momentum(spin)?)
            .into_iter()
            .map(|projection| projection_to_python(py, projection))
            .collect()
    }
}

use laddu_core::{
    AllowedPartialWave, Charge, Isospin, LadduError, OrbitalAngularMomentum, Parity, PartialWave,
    ParticleProperties, RuleSet, SelectionRules, Statistics,
};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyBool},
    IntoPyObjectExt,
};

use self::angular_momentum::{
    angular_momentum_to_python, parse_angular_momentum, parse_orbital_angular_momentum,
    parse_projection, projection_to_python,
};

type PyQuantumNumber = Py<PyAny>;

fn parse_parity(input: &Bound<'_, PyAny>) -> PyResult<Parity> {
    if let Ok(value) = input.extract::<PyParity>() {
        return Ok(value.0);
    }
    if let Ok(value) = input.extract::<String>() {
        return Ok(value.parse()?);
    }
    Err(PyTypeError::new_err(
        "parity must be a Parity or sign string",
    ))
}

fn parse_statistics(input: &Bound<'_, PyAny>) -> PyResult<Statistics> {
    if let Ok(value) = input.extract::<PyStatistics>() {
        return Ok(value.into());
    }
    if let Ok(value) = input.extract::<String>() {
        return match value.to_ascii_lowercase().as_str() {
            "boson" | "bosonic" => Ok(Statistics::Boson),
            "fermion" | "fermionic" => Ok(Statistics::Fermion),
            _ => Err(LadduError::ParseError {
                name: value,
                object: "Statistics".to_string(),
            }
            .into()),
        };
    }
    Err(PyTypeError::new_err(
        "statistics must be a Statistics value or string",
    ))
}

fn parse_charge_input(input: &Bound<'_, PyAny>) -> PyResult<Charge> {
    if let Ok(value) = input.extract::<PyCharge>() {
        return Ok(value.0);
    }
    if input.is_instance_of::<PyBool>() {
        return Err(LadduError::Custom("electric charge cannot be a bool".to_string()).into());
    }
    if let Ok(value) = input.extract::<i32>() {
        return Ok(Charge::try_from(num::rational::Ratio::from_integer(value))?);
    }
    let numerator = input
        .getattr("numerator")
        .and_then(|value| value.extract::<i32>());
    let denominator = input
        .getattr("denominator")
        .and_then(|value| value.extract::<i32>());
    if let (Ok(numerator), Ok(denominator)) = (numerator, denominator) {
        if denominator == 0 {
            return Err(LadduError::Custom(
                "electric charge denominator cannot be zero".to_string(),
            )
            .into());
        }
        return Ok(Charge::try_from(num::rational::Ratio::new(
            numerator,
            denominator,
        ))?);
    }
    if let Ok(value) = input.extract::<f64>() {
        return Ok(Charge::try_from(value)?);
    }
    Err(PyTypeError::new_err(
        "electric charge must be an int, float, fractions.Fraction, or Charge",
    ))
}

fn charge_to_python(py: Python<'_>, charge: Charge) -> PyResult<PyQuantumNumber> {
    let thirds = charge.value();
    if thirds % 3 == 0 {
        Ok((thirds / 3).into_bound_py_any(py)?.unbind())
    } else {
        let fractions = pyo3::types::PyModule::import(py, "fractions")?;
        let fraction = fractions.getattr("Fraction")?;
        Ok(fraction.call1((thirds, 3))?.unbind())
    }
}

#[pyclass(eq, name = "Parity", module = "laddu", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyParity(pub Parity);

#[pymethods]
impl PyParity {
    #[new]
    fn new(value: &str) -> PyResult<Self> {
        Ok(Self(value.parse()?))
    }

    #[staticmethod]
    fn positive() -> Self {
        Self(Parity::Positive)
    }

    #[staticmethod]
    fn negative() -> Self {
        Self(Parity::Negative)
    }

    #[getter]
    fn value(&self) -> i32 {
        self.0.value()
    }

    fn __repr__(&self) -> String {
        format!("Parity('{}')", self.0)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

#[pyclass(eq, eq_int, name = "Statistics", module = "laddu", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyStatistics {
    Boson,
    Fermion,
}

impl From<PyStatistics> for Statistics {
    fn from(value: PyStatistics) -> Self {
        match value {
            PyStatistics::Boson => Self::Boson,
            PyStatistics::Fermion => Self::Fermion,
        }
    }
}

impl From<Statistics> for PyStatistics {
    fn from(value: Statistics) -> Self {
        match value {
            Statistics::Boson => Self::Boson,
            Statistics::Fermion => Self::Fermion,
        }
    }
}

#[pyclass(eq, name = "Charge", module = "laddu", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyCharge(pub Charge);

#[pymethods]
impl PyCharge {
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(parse_charge_input(value)?))
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<PyQuantumNumber> {
        charge_to_python(py, self.0)
    }

    fn __repr__(&self) -> String {
        format!("Charge({})", self.0)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

#[pyclass(eq, name = "Isospin", module = "laddu", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyIsospin(pub Isospin);

#[pymethods]
impl PyIsospin {
    #[new]
    #[pyo3(signature = (isospin, *, projection=None))]
    fn new(isospin: &Bound<'_, PyAny>, projection: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self(Isospin::new(
            parse_angular_momentum(isospin)?,
            projection.map(parse_projection).transpose()?,
        )?))
    }

    #[getter]
    fn isospin(&self, py: Python<'_>) -> PyResult<PyQuantumNumber> {
        angular_momentum_to_python(py, self.0.isospin())
    }

    #[getter]
    fn projection_unchecked(&self, py: Python<'_>) -> PyResult<Option<PyQuantumNumber>> {
        self.0
            .projection
            .map(|projection| projection_to_python(py, projection))
            .transpose()
    }

    #[getter]
    fn projection(&self, py: Python<'_>) -> PyResult<PyQuantumNumber> {
        projection_to_python(py, self.0.projection()?)
    }

    fn __repr__(&self) -> String {
        match self.0.projection {
            Some(projection) => format!("Isospin({}, projection={})", self.0.isospin(), projection),
            None => format!("Isospin({})", self.0.isospin()),
        }
    }
}

#[pyclass(name = "ParticleProperties", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyParticleProperties(pub ParticleProperties);

#[pymethods]
impl PyParticleProperties {
    #[new]
    #[pyo3(signature = (name=None, *, species=None, antiparticle_species=None, self_conjugate=None, spin=None, parity=None, c_parity=None, g_parity=None, charge=None, isospin=None, strangeness=None, charm=None, bottomness=None, topness=None, baryon_number=None, electron_lepton_number=None, muon_lepton_number=None, tau_lepton_number=None, statistics=None))]
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
        charge: Option<&Bound<'_, PyAny>>,
        isospin: Option<PyIsospin>,
        strangeness: Option<i32>,
        charm: Option<i32>,
        bottomness: Option<i32>,
        topness: Option<i32>,
        baryon_number: Option<i32>,
        electron_lepton_number: Option<i32>,
        muon_lepton_number: Option<i32>,
        tau_lepton_number: Option<i32>,
        statistics: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut properties = ParticleProperties::unknown();
        if let Some(name) = name {
            properties = properties.with_name(name);
        }
        if let Some(species) = species {
            properties = properties.with_species(species);
        }
        if let Some(antiparticle_species) = antiparticle_species {
            properties = properties.with_antiparticle_species(antiparticle_species);
        }
        if let Some(self_conjugate) = self_conjugate {
            properties = properties.with_self_conjugate(self_conjugate);
        }
        if let Some(spin) = spin {
            properties = properties.with_spin(parse_angular_momentum(spin)?);
        }
        if let Some(parity) = parity {
            properties = properties.with_parity(parse_parity(parity)?);
        }
        if let Some(c_parity) = c_parity {
            properties = properties.with_c_parity(parse_parity(c_parity)?);
        }
        if let Some(g_parity) = g_parity {
            properties = properties.with_g_parity(parse_parity(g_parity)?);
        }
        if let Some(charge) = charge {
            properties = properties.with_charge(parse_charge_input(charge)?);
        }
        if let Some(isospin) = isospin {
            properties = properties.with_isospin(isospin.0);
        }
        if let Some(strangeness) = strangeness {
            properties = properties.with_strangeness(strangeness);
        }
        if let Some(charm) = charm {
            properties = properties.with_charm(charm);
        }
        if let Some(bottomness) = bottomness {
            properties = properties.with_bottomness(bottomness);
        }
        if let Some(topness) = topness {
            properties = properties.with_topness(topness);
        }
        if let Some(baryon_number) = baryon_number {
            properties = properties.with_baryon_number(baryon_number);
        }
        if let Some(electron_lepton_number) = electron_lepton_number {
            properties = properties.with_electron_lepton_number(electron_lepton_number);
        }
        if let Some(muon_lepton_number) = muon_lepton_number {
            properties = properties.with_muon_lepton_number(muon_lepton_number);
        }
        if let Some(tau_lepton_number) = tau_lepton_number {
            properties = properties.with_tau_lepton_number(tau_lepton_number);
        }
        if let Some(statistics) = statistics {
            properties = properties.with_statistics(parse_statistics(statistics)?)?;
        }
        Ok(Self(properties))
    }

    #[getter]
    fn name(&self) -> PyResult<String> {
        Ok(self.0.name()?)
    }

    #[getter]
    fn name_unchecked(&self) -> Option<String> {
        self.0.name.clone()
    }

    #[getter]
    fn species(&self) -> PyResult<String> {
        Ok(self.0.species()?)
    }

    #[getter]
    fn species_unchecked(&self) -> Option<String> {
        self.0.species.clone()
    }

    #[getter]
    fn antiparticle_species(&self) -> PyResult<String> {
        Ok(self.0.antiparticle_species()?)
    }

    #[getter]
    fn antiparticle_species_unchecked(&self) -> Option<String> {
        self.0.antiparticle_species.clone()
    }

    #[getter]
    fn self_conjugate(&self) -> PyResult<bool> {
        Ok(self.0.self_conjugate()?)
    }

    #[getter]
    fn self_conjugate_unchecked(&self) -> Option<bool> {
        self.0.self_conjugate
    }

    #[getter]
    fn spin(&self, py: Python<'_>) -> PyResult<PyQuantumNumber> {
        angular_momentum_to_python(py, self.0.spin()?)
    }

    #[getter]
    fn spin_unchecked(&self, py: Python<'_>) -> PyResult<Option<PyQuantumNumber>> {
        self.0
            .spin
            .map(|spin| angular_momentum_to_python(py, spin))
            .transpose()
    }

    #[getter]
    fn parity(&self) -> PyResult<PyParity> {
        Ok(PyParity(self.0.parity()?))
    }

    #[getter]
    fn parity_unchecked(&self) -> Option<PyParity> {
        self.0.parity.map(PyParity)
    }

    #[getter]
    fn c_parity(&self) -> PyResult<PyParity> {
        Ok(PyParity(self.0.c_parity()?))
    }

    #[getter]
    fn c_parity_unchecked(&self) -> Option<PyParity> {
        self.0.c_parity.map(PyParity)
    }

    #[getter]
    fn g_parity(&self) -> PyResult<PyParity> {
        Ok(PyParity(self.0.g_parity()?))
    }

    #[getter]
    fn g_parity_unchecked(&self) -> Option<PyParity> {
        self.0.g_parity.map(PyParity)
    }

    #[getter]
    fn charge(&self) -> PyResult<PyCharge> {
        Ok(PyCharge(self.0.charge()?))
    }

    #[getter]
    fn charge_unchecked(&self) -> Option<PyCharge> {
        self.0.charge.map(PyCharge)
    }

    #[getter]
    fn isospin(&self) -> PyResult<PyIsospin> {
        Ok(PyIsospin(self.0.isospin()?))
    }

    #[getter]
    fn isospin_unchecked(&self) -> Option<PyIsospin> {
        self.0.isospin.map(PyIsospin)
    }

    #[getter]
    fn strangeness(&self) -> PyResult<i32> {
        Ok(self.0.strangeness()?)
    }

    #[getter]
    fn strangeness_unchecked(&self) -> Option<i32> {
        self.0.strangeness
    }

    #[getter]
    fn charm(&self) -> PyResult<i32> {
        Ok(self.0.charm()?)
    }

    #[getter]
    fn charm_unchecked(&self) -> Option<i32> {
        self.0.charm
    }

    #[getter]
    fn bottomness(&self) -> PyResult<i32> {
        Ok(self.0.bottomness()?)
    }

    #[getter]
    fn bottomness_unchecked(&self) -> Option<i32> {
        self.0.bottomness
    }

    #[getter]
    fn topness(&self) -> PyResult<i32> {
        Ok(self.0.topness()?)
    }

    #[getter]
    fn topness_unchecked(&self) -> Option<i32> {
        self.0.topness
    }

    #[getter]
    fn baryon_number(&self) -> PyResult<i32> {
        Ok(self.0.baryon_number()?)
    }

    #[getter]
    fn baryon_number_unchecked(&self) -> Option<i32> {
        self.0.baryon_number
    }

    #[getter]
    fn electron_lepton_number(&self) -> PyResult<i32> {
        Ok(self.0.electron_lepton_number()?)
    }

    #[getter]
    fn electron_lepton_number_unchecked(&self) -> Option<i32> {
        self.0.electron_lepton_number
    }

    #[getter]
    fn muon_lepton_number(&self) -> PyResult<i32> {
        Ok(self.0.muon_lepton_number()?)
    }

    #[getter]
    fn muon_lepton_number_unchecked(&self) -> Option<i32> {
        self.0.muon_lepton_number
    }

    #[getter]
    fn tau_lepton_number(&self) -> PyResult<i32> {
        Ok(self.0.tau_lepton_number()?)
    }

    #[getter]
    fn tau_lepton_number_unchecked(&self) -> Option<i32> {
        self.0.tau_lepton_number
    }

    #[getter]
    fn statistics(&self) -> PyResult<PyStatistics> {
        Ok(self.0.statistics()?.into())
    }

    #[getter]
    fn statistics_unchecked(&self) -> Option<PyStatistics> {
        self.0.statistics.map(|s| s.into())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(eq, name = "PartialWave", module = "laddu", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyPartialWave(pub PartialWave);

#[pymethods]
impl PyPartialWave {
    #[new]
    #[pyo3(signature = (*, j, l, s, label=None))]
    fn new(
        j: &Bound<'_, PyAny>,
        l: &Bound<'_, PyAny>,
        s: &Bound<'_, PyAny>,
        label: Option<String>,
    ) -> PyResult<Self> {
        let wave = PartialWave::new(
            parse_angular_momentum(j)?,
            parse_orbital_angular_momentum(l)?,
            parse_angular_momentum(s)?,
        )?;
        Ok(Self(match label {
            Some(label) => wave.with_label(label),
            None => wave,
        }))
    }

    #[getter]
    fn j(&self, py: Python<'_>) -> PyResult<PyQuantumNumber> {
        angular_momentum_to_python(py, self.0.j)
    }

    #[getter]
    fn l(&self) -> u32 {
        self.0.l.value()
    }

    #[getter]
    fn s(&self, py: Python<'_>) -> PyResult<PyQuantumNumber> {
        angular_momentum_to_python(py, self.0.s)
    }

    #[getter]
    fn label(&self) -> String {
        self.0.label.clone()
    }

    fn __repr__(&self) -> String {
        format!("PartialWave('{}')", self.0.label)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

#[pyclass(eq, name = "AllowedPartialWave", module = "laddu", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyAllowedPartialWave(pub AllowedPartialWave);

#[pymethods]
impl PyAllowedPartialWave {
    #[getter]
    fn wave(&self) -> PyPartialWave {
        PyPartialWave(self.0.wave.clone())
    }

    #[getter]
    fn parity(&self) -> Option<PyParity> {
        self.0.parity.map(PyParity)
    }

    #[getter]
    fn c_parity(&self) -> Option<PyParity> {
        self.0.c_parity.map(PyParity)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(eq, name = "RuleSet", module = "laddu", from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyRuleSet(pub RuleSet);

#[pymethods]
impl PyRuleSet {
    #[new]
    fn new() -> Self {
        Self(RuleSet::default())
    }

    #[staticmethod]
    fn angular() -> Self {
        Self(RuleSet::angular())
    }

    #[staticmethod]
    fn strong() -> Self {
        Self(RuleSet::strong())
    }

    #[staticmethod]
    fn electromagnetic() -> Self {
        Self(RuleSet::electromagnetic())
    }

    #[staticmethod]
    fn weak() -> Self {
        Self(RuleSet::weak())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

fn parse_rules(rules: Option<&Bound<'_, PyAny>>) -> PyResult<RuleSet> {
    let Some(rules) = rules else {
        return Ok(RuleSet::strong());
    };
    if let Ok(rules) = rules.extract::<PyRuleSet>() {
        return Ok(rules.0);
    }
    if let Ok(name) = rules.extract::<String>() {
        return match name.to_ascii_lowercase().as_str() {
            "angular" => Ok(RuleSet::angular()),
            "strong" => Ok(RuleSet::strong()),
            "electromagnetic" | "em" => Ok(RuleSet::electromagnetic()),
            "weak" => Ok(RuleSet::weak()),
            _ => Err(LadduError::ParseError {
                name,
                object: "RuleSet".to_string(),
            }
            .into()),
        };
    }
    Err(PyTypeError::new_err(
        "rules must be a RuleSet or preset string",
    ))
}

#[pyclass(name = "SelectionRules", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PySelectionRules(pub SelectionRules);

#[pymethods]
impl PySelectionRules {
    #[new]
    #[pyo3(signature = (*, max_l=6, rules=None))]
    fn new(max_l: u32, rules: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self(SelectionRules {
            max_l: OrbitalAngularMomentum::integer(max_l),
            rules: parse_rules(rules)?,
        }))
    }

    #[staticmethod]
    fn coupled_spins(
        py: Python<'_>,
        spin_1: &Bound<'_, PyAny>,
        spin_2: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyQuantumNumber>> {
        SelectionRules::coupled_spins(
            parse_angular_momentum(spin_1)?,
            parse_angular_momentum(spin_2)?,
        )
        .into_iter()
        .map(|spin| angular_momentum_to_python(py, spin))
        .collect()
    }

    fn allowed_partial_waves(
        &self,
        parent: &PyParticleProperties,
        daughter_1: &PyParticleProperties,
        daughter_2: &PyParticleProperties,
    ) -> Vec<PyAllowedPartialWave> {
        self.0
            .allowed_partial_waves(&parent.0, (&daughter_1.0, &daughter_2.0))
            .into_iter()
            .map(PyAllowedPartialWave)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Return all possible coupled total spins from two daughter spins.
#[pyfunction(name = "coupled_spins")]
pub fn py_coupled_spins(
    py: Python<'_>,
    spin_1: &Bound<'_, PyAny>,
    spin_2: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyQuantumNumber>> {
    PySelectionRules::coupled_spins(py, spin_1, spin_2)
}

/// Generate allowed two-body partial waves.
#[pyfunction(name = "allowed_partial_waves", signature = (parent, daughter_1, daughter_2, *, max_l=6, rules=None))]
pub fn py_allowed_partial_waves(
    parent: &PyParticleProperties,
    daughter_1: &PyParticleProperties,
    daughter_2: &PyParticleProperties,
    max_l: u32,
    rules: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<PyAllowedPartialWave>> {
    Ok(PySelectionRules::new(max_l, rules)?
        .0
        .allowed_partial_waves(&parent.0, (&daughter_1.0, &daughter_2.0))
        .into_iter()
        .map(PyAllowedPartialWave)
        .collect())
}
