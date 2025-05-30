use crate::utils::variables::PyVariable;
use laddu_core::{
    data::{open, open_boosted_to_rest_frame_of, BinnedDataset, Dataset, Event},
    Float,
};
use numpy::PyArray1;
use pyo3::{
    exceptions::{PyIndexError, PyTypeError},
    prelude::*,
};
use std::{path::PathBuf, sync::Arc};

use crate::utils::vectors::{PyVec3, PyVec4};

/// A single event
///
/// Events are composed of a set of 4-momenta of particles in the overall
/// center-of-momentum frame, polarizations or helicities described by 3-vectors, and a
/// weight
///
/// Parameters
/// ----------
/// p4s : list of Vec4
///     4-momenta of each particle in the event in the overall center-of-momentum frame
/// aux: list of Vec3
///     3-vectors describing auxiliary data for each particle given in `p4s`
/// weight : float
///     The weight associated with this event
/// rest_frame_indices : list of int, optional
///     If supplied, the event will be boosted to the rest frame of the 4-momenta at the
///     given indices
///
#[pyclass(name = "Event", module = "laddu")]
#[derive(Clone)]
pub struct PyEvent(pub Arc<Event>);

#[pymethods]
impl PyEvent {
    #[new]
    #[pyo3(signature = (p4s, aux, weight, *, rest_frame_indices=None))]
    fn new(
        p4s: Vec<PyVec4>,
        aux: Vec<PyVec3>,
        weight: Float,
        rest_frame_indices: Option<Vec<usize>>,
    ) -> Self {
        let event = Event {
            p4s: p4s.into_iter().map(|arr| arr.0).collect(),
            aux: aux.into_iter().map(|arr| arr.0).collect(),
            weight,
        };
        if let Some(indices) = rest_frame_indices {
            Self(Arc::new(event.boost_to_rest_frame_of(indices)))
        } else {
            Self(Arc::new(event))
        }
    }
    fn __str__(&self) -> String {
        self.0.to_string()
    }
    /// The list of 4-momenta for each particle in the event
    ///
    #[getter]
    fn get_p4s(&self) -> Vec<PyVec4> {
        self.0.p4s.iter().map(|p4| PyVec4(*p4)).collect()
    }
    /// The list of 3-vectors describing the auxiliary data of particles in
    /// the event
    ///
    #[getter]
    fn get_aux(&self) -> Vec<PyVec3> {
        self.0.aux.iter().map(|eps_vec| PyVec3(*eps_vec)).collect()
    }
    /// The weight of this event relative to others in a Dataset
    ///
    #[getter]
    fn get_weight(&self) -> Float {
        self.0.weight
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
        PyVec4(self.0.get_p4_sum(indices))
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
        PyEvent(Arc::new(self.0.boost_to_rest_frame_of(indices)))
    }
}

/// A set of Events
///
/// Datasets can be created from lists of Events or by using the provided ``laddu.open`` function
///
/// Datasets can also be indexed directly to access individual Events
///
/// Parameters
/// ----------
/// events : list of Event
///
/// See Also
/// --------
/// laddu.open
///
#[pyclass(name = "DatasetBase", module = "laddu", subclass)]
#[derive(Clone)]
pub struct PyDataset(pub Arc<Dataset>);

#[pymethods]
impl PyDataset {
    #[new]
    fn new(events: Vec<PyEvent>) -> Self {
        Self(Arc::new(Dataset::new(
            events.into_iter().map(|event| event.0).collect(),
        )))
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
    /// Get the weighted number of Events in the Dataset
    ///
    /// Returns
    /// -------
    /// n_events : float
    ///     The sum of all Event weights
    ///
    #[getter]
    fn n_events_weighted(&self) -> Float {
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
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
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
            .map(|rust_event| PyEvent(rust_event.clone()))
            .collect()
    }
    fn __getitem__(&self, index: usize) -> PyEvent {
        PyEvent(Arc::new(self.0[index].clone()))
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
        range: (Float, Float),
    ) -> PyResult<PyBinnedDataset> {
        let py_variable = variable.extract::<PyVariable>()?;
        Ok(PyBinnedDataset(self.0.bin_by(py_variable, bins, range)))
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
    /// Generate a new bootstrapped Dataset by randomly resampling the original with replacement
    ///
    /// The new Dataset is resampled with a random generator seeded by the provided `seed`
    /// This is the same as [`bootstrap`], but the event weights are used in the
    /// sample to attempt to maintain the same weighted sum of events.
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
    fn weighted_bootstrap(&self, seed: usize) -> PyDataset {
        PyDataset(self.0.weighted_bootstrap(seed))
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
    fn range(&self) -> (Float, Float) {
        self.0.range()
    }
    /// The edges of each bin in the BinnedDataset
    ///
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, &self.0.edges())
    }
    fn __getitem__(&self, index: usize) -> PyResult<PyDataset> {
        self.0
            .get(index)
            .ok_or(PyIndexError::new_err("index out of range"))
            .map(|rust_dataset| PyDataset(rust_dataset.clone()))
    }
}

/// Open a Dataset from a file
///
/// Arguments
/// ---------
/// path : str or Path
///     The path to the file
/// rest_frame_indices : list of int, optional
///     If supplied, the dataset will be boosted to the rest frame of the 4-momenta at the
///     given indices
///
///
/// Returns
/// -------
/// Dataset
///
/// Raises
/// ------
/// IOError
///     If the file could not be read
///
/// Warnings
/// --------
/// This method will panic/fail if the columns do not have the correct names or data types.
/// There is currently no way to make this nicer without a large performance dip (if you find a
/// way, please open a PR).
///
/// Notes
/// -----
/// Data should be stored in Parquet format with each column being filled with 32-bit floats
///
/// Valid/required column names have the following formats:
///
/// ``p4_{particle index}_{E|Px|Py|Pz}`` (four-momentum components for each particle)
///
/// ``aux_{particle index}_{x|y|z}`` (auxiliary vectors for each particle)
///
/// ``weight`` (the weight of the Event)
///
/// For example, the four-momentum of the 0th particle in the event would be stored in columns
/// with the names ``p4_0_E``, ``p4_0_Px``, ``p4_0_Py``, and ``p4_0_Pz``. That particle's
/// polarization could be stored in the columns ``aux_0_x``, ``aux_0_y``, and ``aux_0_z``. This
/// could continue for an arbitrary number of particles. The ``weight`` column is always
/// required.
///
#[pyfunction(name = "open", signature = (path, *, rest_frame_indices=None))]
pub fn py_open(path: Bound<PyAny>, rest_frame_indices: Option<Vec<usize>>) -> PyResult<PyDataset> {
    let path_str = if let Ok(s) = path.extract::<String>() {
        Ok(s)
    } else if let Ok(pathbuf) = path.extract::<PathBuf>() {
        Ok(pathbuf.to_string_lossy().into_owned())
    } else {
        Err(PyTypeError::new_err("Expected str or Path"))
    }?;
    if let Some(indices) = rest_frame_indices {
        Ok(PyDataset(open_boosted_to_rest_frame_of(path_str, indices)?))
    } else {
        Ok(PyDataset(open(path_str)?))
    }
}
