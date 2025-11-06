use crate::utils::variables::{PyVariable, PyVariableExpression};
use laddu_core::{
    data::{BinnedDataset, Dataset, DatasetMetadata, Event, EventData},
    DatasetReadOptions,
};
use numpy::PyArray1;
use pyo3::{
    exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyIterator, PyList},
    IntoPyObjectExt,
};
use std::{path::PathBuf, sync::Arc};

use crate::utils::vectors::PyVec4;

/// A single event
///
/// Events are composed of a set of 4-momenta of particles in the overall
/// center-of-momentum frame, optional auxiliary scalars (e.g. polarization magnitude or angle),
/// and a weight.
///
/// Parameters
/// ----------
/// p4s : list of Vec4
///     4-momenta of each particle in the event in the overall center-of-momentum frame
/// aux: list of float
///     Scalar auxiliary data associated with the event
/// weight : float
///     The weight associated with this event
/// rest_frame_of : list of str, optional
///     If supplied, the event will be boosted to the rest frame of the named four-momenta
/// p4_names : list of str, optional
///     Human-readable aliases for each four-momentum. Providing names enables name-based
///     lookups when evaluating variables.
/// aux_names : list of str, optional
///     Aliases for auxiliary scalars corresponding to ``aux``.
///
#[pyclass(name = "Event", module = "laddu")]
#[derive(Clone)]
pub struct PyEvent {
    pub event: Event,
    has_metadata: bool,
}

#[pymethods]
impl PyEvent {
    #[new]
    #[pyo3(signature = (p4s, aux, weight, *, rest_frame_of=None, p4_names=None, aux_names=None))]
    fn new(
        p4s: Vec<PyVec4>,
        aux: Vec<f64>,
        weight: f64,
        rest_frame_of: Option<Vec<String>>,
        p4_names: Option<Vec<String>>,
        aux_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut event = EventData {
            p4s: p4s.into_iter().map(|arr| arr.0).collect(),
            aux,
            weight,
        };
        let (metadata, metadata_provided) = if p4_names.is_some() || aux_names.is_some() {
            let p4_names = p4_names.unwrap_or_default();
            let aux_names = aux_names.unwrap_or_default();
            (
                Arc::new(DatasetMetadata::new(p4_names, aux_names).map_err(PyErr::from)?),
                true,
            )
        } else {
            (Arc::new(DatasetMetadata::empty()), false)
        };
        if let Some(names) = rest_frame_of {
            if !metadata_provided {
                return Err(PyValueError::new_err(
                    "`rest_frame_of` requires specifying `p4_names` to resolve particle names",
                ));
            }
            let indices = names
                .iter()
                .map(|name| {
                    metadata.p4_index(name).ok_or_else(|| {
                        PyKeyError::new_err(format!("Unknown particle name '{name}'"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            event = event.boost_to_rest_frame_of(indices);
        }
        let event = Event::new(Arc::new(event), metadata);
        Ok(Self {
            event,
            has_metadata: metadata_provided,
        })
    }
    fn __str__(&self) -> String {
        self.event.data().to_string()
    }
    /// The list of 4-momenta for each particle in the event
    ///
    #[getter]
    fn get_p4s(&self) -> Vec<PyVec4> {
        self.event.p4s().iter().map(|p4| PyVec4(*p4)).collect()
    }
    /// The list of auxiliary scalar values associated with the event
    ///
    #[getter]
    #[pyo3(name = "aux_values")]
    fn aux_values_prop(&self) -> Vec<f64> {
        self.event.aux_values().to_vec()
    }
    /// The weight of this event relative to others in a Dataset
    ///
    #[getter]
    fn get_weight(&self) -> f64 {
        self.event.weight()
    }
    /// Get the sum of the four-momenta within the event at the given indices
    ///
    /// Parameters
    /// ----------
    /// names : list of str
    ///     The names of the four-momenta to sum
    ///
    /// Returns
    /// -------
    /// Vec4
    ///     The result of summing the given four-momenta
    ///
    fn get_p4_sum(&self, names: Vec<String>) -> PyResult<PyVec4> {
        let indices = self.resolve_p4_indices(&names)?;
        Ok(PyVec4(self.event.data().get_p4_sum(indices)))
    }
    /// Boost all the four-momenta in the event to the rest frame of the given set of
    /// four-momenta by indices.
    ///
    /// Parameters
    /// ----------
    /// names : list of str
    ///     The names of the four-momenta whose rest frame should be used for the boost
    ///
    /// Returns
    /// -------
    /// Event
    ///     The boosted event
    ///
    pub fn boost_to_rest_frame_of(&self, names: Vec<String>) -> PyResult<Self> {
        let indices = self.resolve_p4_indices(&names)?;
        let boosted = self.event.data().boost_to_rest_frame_of(indices);
        Ok(Self {
            event: Event::new(Arc::new(boosted), self.event.metadata_arc()),
            has_metadata: self.has_metadata,
        })
    }
    /// Get the value of a Variable on the given Event
    ///
    /// Parameters
    /// ----------
    /// variable : {laddu.Mass, laddu.CosTheta, laddu.Phi, laddu.PolAngle, laddu.PolMagnitude, laddu.Mandelstam}
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Notes
    /// -----
    /// Variables that rely on particle names require the event to carry metadata. Provide
    /// ``p4_names``/``aux_names`` when constructing the event or evaluate variables through a
    /// ``laddu.Dataset`` to ensure the metadata is available.
    ///
    fn evaluate(&self, variable: Bound<'_, PyAny>) -> PyResult<f64> {
        let mut variable = variable.extract::<PyVariable>()?;
        if !self.has_metadata {
            return Err(PyValueError::new_err(
                "Cannot evaluate variable on an Event without associated metadata. Construct the Event with `p4_names`/`aux_names` or evaluate through a Dataset.",
            ));
        }
        variable.bind_in_place(self.event.metadata())?;
        let event_arc = self.event.data_arc();
        variable.evaluate_event(&event_arc)
    }

    /// Retrieve a four-momentum by name (if present).
    fn p4(&self, name: &str) -> PyResult<Option<PyVec4>> {
        self.ensure_metadata()?;
        Ok(self.event.p4(name).copied().map(PyVec4))
    }

    /// Retrieve an auxiliary scalar by name (if present).
    fn aux(&self, name: &str) -> PyResult<Option<f64>> {
        self.ensure_metadata()?;
        Ok(self.event.aux(name))
    }

    /// Retrieve either a four-momentum or auxiliary scalar by name, returning ``None`` when not found.
    fn get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        self.ensure_metadata()?;
        if let Some(p4) = self.event.p4(name) {
            let obj = PyVec4(*p4).into_pyobject(py)?.into_any().unbind();
            return Ok(Some(obj));
        }
        if let Some(value) = self.event.aux(name) {
            let obj = value.into_pyobject(py)?.into_any().unbind();
            return Ok(Some(obj));
        }
        Ok(None)
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let name = key.extract::<String>()?;
        if !self.has_metadata {
            return Err(PyValueError::new_err(
                "Event has no associated metadata; indexing by name is unavailable",
            ));
        }
        if let Some(p4) = self.event.p4(&name) {
            return PyVec4(*p4).into_bound_py_any(py);
        }
        if let Some(value) = self.event.aux(&name) {
            return value.into_bound_py_any(py);
        }
        Err(PyKeyError::new_err(format!(
            "Unknown particle or auxiliary name '{name}'",
        )))
    }
}

impl PyEvent {
    fn ensure_metadata(&self) -> PyResult<&DatasetMetadata> {
        if !self.has_metadata {
            Err(PyValueError::new_err(
                "Event has no associated metadata for name-based operations",
            ))
        } else {
            Ok(self.event.metadata())
        }
    }

    fn resolve_p4_indices(&self, names: &[String]) -> PyResult<Vec<usize>> {
        let metadata = self.ensure_metadata()?;
        names
            .iter()
            .map(|name| {
                metadata
                    .p4_index(name)
                    .ok_or_else(|| PyKeyError::new_err(format!("Unknown particle name '{name}'")))
            })
            .collect()
    }

    pub(crate) fn metadata_opt(&self) -> Option<&DatasetMetadata> {
        self.has_metadata.then(|| self.event.metadata())
    }
}

/// A set of Events
///
/// Datasets can be created from lists of Events or by using :meth:`laddu.Dataset.open`
///
/// Datasets can also be indexed directly to access individual Events
///
/// Parameters
/// ----------
/// events : list of Event
/// p4_names : list of str, optional
///     Names assigned to each four-momentum; enables name-based lookups if provided.
/// aux_names : list of str, optional
///     Names for auxiliary scalars stored alongside the events.
///
/// See Also
/// --------
/// laddu.Dataset.open
///
#[pyclass(name = "DatasetBase", module = "laddu", subclass)]
#[derive(Clone)]
pub struct PyDataset(pub Arc<Dataset>);

#[pymethods]
impl PyDataset {
    #[new]
    #[pyo3(signature = (events, *, p4_names=None, aux_names=None))]
    fn new(
        events: Vec<PyEvent>,
        p4_names: Option<Vec<String>>,
        aux_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let inferred_metadata = events
            .iter()
            .find_map(|event| event.has_metadata.then(|| event.event.metadata_arc()));

        let use_explicit_names = p4_names.is_some() || aux_names.is_some();
        let metadata = if use_explicit_names {
            let p4_names = p4_names.unwrap_or_default();
            let aux_names = aux_names.unwrap_or_default();
            Some(Arc::new(
                DatasetMetadata::new(p4_names, aux_names).map_err(PyErr::from)?,
            ))
        } else {
            inferred_metadata
        };

        let events: Vec<Arc<EventData>> = events
            .into_iter()
            .map(|event| event.event.data_arc())
            .collect();
        let dataset = if let Some(metadata) = metadata {
            Dataset::new_with_metadata(events, metadata)
        } else {
            Dataset::new(events)
        };
        Ok(Self(Arc::new(dataset)))
    }

    /// Open a Dataset from a Parquet file.
    ///
    /// Parameters
    /// ----------
    /// path : str or Path
    ///     The path to the Parquet file.
    /// p4s : list[str], optional
    ///     Particle identifiers corresponding to ``*_px``, ``*_py``, ``*_pz``, ``*_e`` columns.
    /// aux : list[str], optional
    ///     Auxiliary scalar column names copied verbatim in order.
    /// boost_to_restframe_of : list[str], optional
    ///     Names of particles whose rest frame should be used to boost each event.
    ///
    /// Notes
    /// -----
    /// If `p4s` or `aux` are not provided, they will be inferred from the column names. If all of
    /// the valid suffixes are provided for a particle, the corresponding columns will be read as a
    /// four-momentum, otherwise they will be read as auxiliary scalars.
    #[staticmethod]
    #[pyo3(signature = (path, *, p4s=None, aux=None, boost_to_restframe_of=None))]
    fn open(
        path: Bound<PyAny>,
        p4s: Option<Vec<String>>,
        aux: Option<Vec<String>>,
        boost_to_restframe_of: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let path_str = if let Ok(s) = path.extract::<String>() {
            Ok(s)
        } else if let Ok(pathbuf) = path.extract::<PathBuf>() {
            Ok(pathbuf.to_string_lossy().into_owned())
        } else {
            Err(PyTypeError::new_err("Expected str or Path"))
        }?;

        let mut read_options = DatasetReadOptions::default();
        if let Some(p4s) = p4s {
            read_options = read_options.p4_names(p4s);
        }
        if let Some(aux) = aux {
            read_options = read_options.aux_names(aux);
        }
        if let Some(boost_to_restframe_of) = boost_to_restframe_of {
            read_options = read_options.boost_to_restframe_of(boost_to_restframe_of);
        }
        let dataset = Dataset::open(&path_str, &read_options)?;

        Ok(Self(dataset))
    }
    fn __len__(&self) -> usize {
        self.0.n_events()
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let list_bound = PyList::empty(py);
        for event in slf.events() {
            list_bound.append(Py::new(py, event)?)?;
        }
        let iterator = PyIterator::from_object(list_bound.as_any())?;
        Ok(iterator.into())
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataset> {
        if let Ok(other_ds) = other.extract::<PyRef<PyDataset>>() {
            Ok(PyDataset(Arc::new(self.0.as_ref() + other_ds.0.as_ref())))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataset> {
        if let Ok(other_ds) = other.extract::<PyRef<PyDataset>>() {
            Ok(PyDataset(Arc::new(other_ds.0.as_ref() + self.0.as_ref())))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    /// Get the number of Events in the Dataset
    ///
    /// Returns
    /// -------
    /// n_events : int
    ///     The number of Events
    ///
    #[getter]
    fn n_events(&self) -> usize {
        self.0.n_events()
    }
    /// Particle names used to construct four-momenta when loading from a Parquet file.
    #[getter]
    fn p4_names(&self) -> Vec<String> {
        self.0.p4_names().to_vec()
    }
    /// Auxiliary scalar names associated with this Dataset.
    #[getter]
    fn aux_names(&self) -> Vec<String> {
        self.0.aux_names().to_vec()
    }
    /// Get the weighted number of Events in the Dataset
    ///
    /// Returns
    /// -------
    /// n_events : float
    ///     The sum of all Event weights
    ///
    #[getter]
    fn n_events_weighted(&self) -> f64 {
        self.0.n_events_weighted()
    }
    /// The weights associated with the Dataset
    ///
    /// Returns
    /// -------
    /// weights : array_like
    ///     A ``numpy`` array of Event weights
    ///
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.0.weights())
    }
    /// The internal list of Events stored in the Dataset
    ///
    /// Returns
    /// -------
    /// events : list of Event
    ///     The Events in the Dataset
    ///
    #[getter]
    fn events(&self) -> Vec<PyEvent> {
        self.0
            .events
            .iter()
            .map(|rust_event| PyEvent {
                event: rust_event.clone(),
                has_metadata: true,
            })
            .collect()
    }
    /// Retrieve a four-momentum by particle name for the event at ``index``.
    fn p4_by_name(&self, index: usize, name: &str) -> PyResult<PyVec4> {
        self.0
            .p4_by_name(index, name)
            .map(PyVec4)
            .ok_or_else(|| PyKeyError::new_err(format!("Unknown particle name '{name}'")))
    }
    /// Retrieve an auxiliary scalar by name for the event at ``index``.
    fn aux_by_name(&self, index: usize, name: &str) -> PyResult<f64> {
        self.0
            .aux_by_name(index, name)
            .ok_or_else(|| PyKeyError::new_err(format!("Unknown auxiliary name '{name}'")))
    }
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(value) = self.evaluate(py, index.clone()) {
            value.into_bound_py_any(py)
        } else if let Ok(index) = index.extract::<usize>() {
            PyEvent {
                event: self.0[index].clone(),
                has_metadata: true,
            }
            .into_bound_py_any(py)
        } else {
            Err(PyTypeError::new_err(
                "Unsupported index type (int or Variable)",
            ))
        }
    }
    /// Separates a Dataset into histogram bins by a Variable value
    ///
    /// Parameters
    /// ----------
    /// variable : {laddu.Mass, laddu.CosTheta, laddu.Phi, laddu.PolAngle, laddu.PolMagnitude, laddu.Mandelstam}
    ///     The Variable by which each Event is binned
    /// bins : int
    ///     The number of equally-spaced bins
    /// range : tuple[float, float]
    ///     The minimum and maximum bin edges
    ///
    /// Returns
    /// -------
    /// datasets : BinnedDataset
    ///     A pub structure that holds a list of Datasets binned by the given `variable`
    ///
    /// See Also
    /// --------
    /// laddu.Mass
    /// laddu.CosTheta
    /// laddu.Phi
    /// laddu.PolAngle
    /// laddu.PolMagnitude
    /// laddu.Mandelstam
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the given `variable` is not a valid variable
    ///
    #[pyo3(signature = (variable, bins, range))]
    fn bin_by(
        &self,
        variable: Bound<'_, PyAny>,
        bins: usize,
        range: (f64, f64),
    ) -> PyResult<PyBinnedDataset> {
        let py_variable = variable.extract::<PyVariable>()?;
        let bound_variable = py_variable.bound(self.0.metadata())?;
        Ok(PyBinnedDataset(self.0.bin_by(
            bound_variable,
            bins,
            range,
        )?))
    }
    /// Filter the Dataset by a given VariableExpression, selecting events for which the expression returns ``True``.
    ///
    /// Parameters
    /// ----------
    /// expression : VariableExpression
    ///     The expression with which to filter the Dataset
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     The filtered Dataset
    ///
    pub fn filter(&self, expression: &PyVariableExpression) -> PyResult<PyDataset> {
        Ok(PyDataset(
            self.0.filter(&expression.0).map_err(PyErr::from)?,
        ))
    }
    /// Generate a new bootstrapped Dataset by randomly resampling the original with replacement
    ///
    /// The new Dataset is resampled with a random generator seeded by the provided `seed`
    ///
    /// Parameters
    /// ----------
    /// seed : int
    ///     The random seed used in the resampling process
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     A bootstrapped Dataset
    ///
    fn bootstrap(&self, seed: usize) -> PyDataset {
        PyDataset(self.0.bootstrap(seed))
    }
    /// Boost all the four-momenta in all events to the rest frame of the given set of
    /// named four-momenta.
    ///
    /// Parameters
    /// ----------
    /// names : list of str
    ///     The names of the four-momenta defining the rest frame
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     The boosted dataset
    ///
    pub fn boost_to_rest_frame_of(&self, names: Vec<String>) -> PyDataset {
        PyDataset(self.0.boost_to_rest_frame_of(&names))
    }
    /// Get the value of a Variable over every event in the Dataset.
    ///
    /// Parameters
    /// ----------
    /// variable : {laddu.Mass, laddu.CosTheta, laddu.Phi, laddu.PolAngle, laddu.PolMagnitude, laddu.Mandelstam}
    ///
    /// Returns
    /// -------
    /// values : array_like
    ///
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        variable: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let variable = variable.extract::<PyVariable>()?;
        let bound_variable = variable.bound(self.0.metadata())?;
        let values = self.0.evaluate(&bound_variable).map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, values))
    }
}

/// A collection of Datasets binned by a Variable
///
/// BinnedDatasets can be indexed directly to access the underlying Datasets by bin
///
/// See Also
/// --------
/// laddu.Dataset.bin_by
///
#[pyclass(name = "BinnedDataset", module = "laddu")]
pub struct PyBinnedDataset(BinnedDataset);

#[pymethods]
impl PyBinnedDataset {
    fn __len__(&self) -> usize {
        self.0.n_bins()
    }
    /// The number of bins in the BinnedDataset
    ///
    #[getter]
    fn n_bins(&self) -> usize {
        self.0.n_bins()
    }
    /// The minimum and maximum values of the binning Variable used to create this BinnedDataset
    ///
    #[getter]
    fn range(&self) -> (f64, f64) {
        self.0.range()
    }
    /// The edges of each bin in the BinnedDataset
    ///
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.0.edges())
    }
    fn __getitem__(&self, index: usize) -> PyResult<PyDataset> {
        self.0
            .get(index)
            .ok_or(PyIndexError::new_err("index out of range"))
            .map(|rust_dataset| PyDataset(rust_dataset.clone()))
    }
}
