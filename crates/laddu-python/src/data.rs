use crate::utils::variables::{PyVariable, PyVariableExpression};
use laddu_core::data::{BinnedDataset, Dataset, DatasetMetadata, EventData};
use numpy::PyArray1;
use pyo3::{
    exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError},
    prelude::*,
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
/// rest_frame_indices : list of int, optional
///     If supplied, the event will be boosted to the rest frame of the 4-momenta at the
///     given indices
/// p4_names : list of str, optional
///     Human-readable aliases for each four-momentum. Providing names enables name-based
///     lookups when evaluating variables.
/// aux_names : list of str, optional
///     Aliases for auxiliary scalars corresponding to ``aux``.
///
#[pyclass(name = "Event", module = "laddu")]
#[derive(Clone)]
pub struct PyEvent {
    pub event: Arc<EventData>,
    pub metadata: Option<Arc<DatasetMetadata>>,
}

#[pymethods]
impl PyEvent {
    #[new]
    #[pyo3(signature = (p4s, aux, weight, *, rest_frame_indices=None, p4_names=None, aux_names=None))]
    fn new(
        p4s: Vec<PyVec4>,
        aux: Vec<f64>,
        weight: f64,
        rest_frame_indices: Option<Vec<usize>>,
        p4_names: Option<Vec<String>>,
        aux_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let event = EventData {
            p4s: p4s.into_iter().map(|arr| arr.0).collect(),
            aux,
            weight,
        };
        let metadata = if let Some(p4_names) = p4_names {
            let aux_names = aux_names.unwrap_or_default();
            Some(Arc::new(
                DatasetMetadata::new(p4_names, aux_names).map_err(PyErr::from)?,
            ))
        } else {
            None
        };
        let event = if let Some(indices) = rest_frame_indices {
            event.boost_to_rest_frame_of(indices)
        } else {
            event
        };
        Ok(Self {
            event: Arc::new(event),
            metadata,
        })
    }
    fn __str__(&self) -> String {
        self.event.to_string()
    }
    /// The list of 4-momenta for each particle in the event
    ///
    #[getter]
    fn get_p4s(&self) -> Vec<PyVec4> {
        self.event.p4s.iter().map(|p4| PyVec4(*p4)).collect()
    }
    /// The list of auxiliary scalar values associated with the event
    ///
    #[getter]
    fn get_aux(&self) -> Vec<f64> {
        self.event.aux.clone()
    }
    /// The weight of this event relative to others in a Dataset
    ///
    #[getter]
    fn get_weight(&self) -> f64 {
        self.event.weight
    }
    /// Get the sum of the four-momenta within the event at the given indices
    ///
    /// Parameters
    /// ----------
    /// indices : list of int
    ///     The indices of the four-momenta to sum
    ///
    /// Returns
    /// -------
    /// Vec4
    ///     The result of summing the given four-momenta
    ///
    fn get_p4_sum(&self, indices: Vec<usize>) -> PyVec4 {
        PyVec4(self.event.get_p4_sum(indices))
    }
    /// Boost all the four-momenta in the event to the rest frame of the given set of
    /// four-momenta by indices.
    ///
    /// Parameters
    /// ----------
    /// indices : list of int
    ///     The indices of the four-momenta to sum
    ///
    /// Returns
    /// -------
    /// Event
    ///     The boosted event
    ///
    pub fn boost_to_rest_frame_of(&self, indices: Vec<usize>) -> Self {
        Self {
            event: Arc::new(self.event.boost_to_rest_frame_of(indices)),
            metadata: self.metadata.clone(),
        }
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
        if let Some(metadata) = &self.metadata {
            variable.bind_in_place(metadata.as_ref())?;
        } else {
            return Err(PyValueError::new_err(
                "Cannot evaluate variable on an Event without associated metadata. Construct the Event with `p4_names`/`aux_names` or evaluate through a Dataset.",
            ));
        }
        variable.evaluate_event(&self.event)
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
            .filter_map(|event| event.metadata.clone())
            .next();
        let metadata = if let Some(p4_names) = p4_names {
            let aux_names = aux_names.unwrap_or_default();
            Some(Arc::new(
                DatasetMetadata::new(p4_names, aux_names).map_err(PyErr::from)?,
            ))
        } else if let Some(meta) = inferred_metadata {
            Some(meta)
        } else {
            None
        };

        let events: Vec<Arc<EventData>> = events.into_iter().map(|event| event.event).collect();
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
    /// p4s : list[str]
    ///     Particle identifiers corresponding to ``*_px``, ``*_py``, ``*_pz``, ``*_e`` columns.
    /// aux : list[str]
    ///     Auxiliary scalar column names copied verbatim in order.
    /// boost_to_restframe_of : list[str], optional
    ///     Names of particles whose rest frame should be used to boost each event.
    #[staticmethod]
    #[pyo3(signature = (path, *, p4s, aux, boost_to_restframe_of=None))]
    fn open(
        path: Bound<PyAny>,
        p4s: Vec<String>,
        aux: Vec<String>,
        boost_to_restframe_of: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let path_str = if let Ok(s) = path.extract::<String>() {
            Ok(s)
        } else if let Ok(pathbuf) = path.extract::<PathBuf>() {
            Ok(pathbuf.to_string_lossy().into_owned())
        } else {
            Err(PyTypeError::new_err("Expected str or Path"))
        }?;

        let dataset = if let Some(boost) = boost_to_restframe_of {
            Dataset::open_boosted(path_str, &p4s, &aux, &boost)?
        } else {
            Dataset::open(path_str, &p4s, &aux)?
        };

        Ok(Self(dataset))
    }
    fn __len__(&self) -> usize {
        self.0.n_events()
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
        let metadata = self.0.metadata_arc();
        self.0
            .events
            .iter()
            .map(|rust_event| PyEvent {
                event: rust_event.clone(),
                metadata: Some(metadata.clone()),
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
            let metadata = self.0.metadata_arc();
            PyEvent {
                event: Arc::new(self.0[index].clone()),
                metadata: Some(metadata),
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
        let bound_variable = py_variable.bound(self.0.metadata()).map_err(PyErr::from)?;
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
    /// four-momenta by indices.
    ///
    /// Parameters
    /// ----------
    /// indices : list of int
    ///     The indices of the four-momenta to sum
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     The boosted dataset
    ///
    pub fn boost_to_rest_frame_of(&self, indices: Vec<usize>) -> PyDataset {
        PyDataset(self.0.boost_to_rest_frame_of(indices))
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
        let bound_variable = variable.bound(self.0.metadata()).map_err(PyErr::from)?;
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
