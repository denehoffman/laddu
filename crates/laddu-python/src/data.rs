use crate::utils::variables::{PyVariable, PyVariableExpression};
use laddu_core::{
    data::{
        BinnedDataset, Dataset, DatasetMetadata, DatasetWriteOptions, Event, EventData,
        FloatPrecision,
    },
    utils::variables::IntoP4Selection,
    DatasetReadOptions,
};
use numpy::PyArray1;
use pyo3::{
    exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
    IntoPyObjectExt,
};
use std::{path::PathBuf, sync::Arc};

use crate::utils::vectors::PyVec4;

fn parse_aliases(aliases: Option<Bound<'_, PyDict>>) -> PyResult<Vec<(String, Vec<String>)>> {
    let Some(aliases) = aliases else {
        return Ok(Vec::new());
    };

    let mut parsed = Vec::new();
    for (key, value) in aliases.iter() {
        let alias_name = key.extract::<String>()?;
        let selection = if let Ok(single) = value.extract::<String>() {
            vec![single]
        } else {
            let seq = value.extract::<Vec<String>>().map_err(|_| {
                PyTypeError::new_err("Alias values must be a string or a sequence of strings")
            })?;
            if seq.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "Alias '{alias_name}' must reference at least one particle",
                )));
            }
            seq
        };
        parsed.push((alias_name, selection));
    }

    Ok(parsed)
}

fn parse_dataset_path(path: Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = path.extract::<String>() {
        Ok(s)
    } else if let Ok(pathbuf) = path.extract::<PathBuf>() {
        Ok(pathbuf.to_string_lossy().into_owned())
    } else {
        Err(PyTypeError::new_err("Expected str or Path"))
    }
}

fn parse_precision_arg(value: Option<&str>) -> PyResult<FloatPrecision> {
    match value.map(|v| v.to_ascii_lowercase()) {
        None => Ok(FloatPrecision::F64),
        Some(name) if name == "f64" || name == "float64" || name == "double" => {
            Ok(FloatPrecision::F64)
        }
        Some(name) if name == "f32" || name == "float32" || name == "float" => {
            Ok(FloatPrecision::F32)
        }
        Some(other) => Err(PyValueError::new_err(format!(
            "Unsupported precision '{other}' (expected 'f64' or 'f32')"
        ))),
    }
}

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
/// p4_names : list of str, optional
///     Human-readable aliases for each four-momentum. Providing names enables name-based
///     lookups when evaluating variables.
/// aux_names : list of str, optional
///     Aliases for auxiliary scalars corresponding to ``aux``.
/// aliases : dict of {str: str or list[str]}, optional
///     Additional particle identifiers that reference one or more entries from ``p4_names``.
///
/// Examples
/// --------
/// >>> from laddu import Event, Vec3  # doctest: +SKIP
/// >>> event = Event(  # doctest: +SKIP
/// ...     [Vec3(0.0, 0.0, 1.0).with_mass(0.0), Vec3(0.0, 0.0, 1.0).with_mass(0.0)],
/// ...     [],
/// ...     1.0,
/// ...     p4_names=['kshort1', 'kshort2'],
/// ...     aliases={'pair': ['kshort1', 'kshort2']},
/// ... )
/// >>> event.p4('pair')  # doctest: +SKIP
/// Vec4(px=0.0, py=0.0, pz=2.0, e=2.0)
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
    #[pyo3(signature = (p4s, aux, weight, *, p4_names=None, aux_names=None, aliases=None))]
    fn new(
        p4s: Vec<PyVec4>,
        aux: Vec<f64>,
        weight: f64,
        p4_names: Option<Vec<String>>,
        aux_names: Option<Vec<String>>,
        aliases: Option<Bound<PyDict>>,
    ) -> PyResult<Self> {
        let event = EventData {
            p4s: p4s.into_iter().map(|arr| arr.0).collect(),
            aux,
            weight,
        };
        let aliases = parse_aliases(aliases)?;

        let missing_p4_names = p4_names
            .as_ref()
            .map(|names| names.is_empty())
            .unwrap_or(true);

        if !aliases.is_empty() && missing_p4_names {
            return Err(PyValueError::new_err(
                "`aliases` requires `p4_names` so selections can be resolved",
            ));
        }

        let metadata_provided = p4_names.is_some() || aux_names.is_some() || !aliases.is_empty();
        let metadata = if metadata_provided {
            let p4_names = p4_names.unwrap_or_default();
            let aux_names = aux_names.unwrap_or_default();
            let mut metadata = DatasetMetadata::new(p4_names, aux_names).map_err(PyErr::from)?;
            if !aliases.is_empty() {
                metadata
                    .add_p4_aliases(
                        aliases.into_iter().map(|(alias_name, selection)| {
                            (alias_name, selection.into_selection())
                        }),
                    )
                    .map_err(PyErr::from)?;
            }
            Arc::new(metadata)
        } else {
            Arc::new(DatasetMetadata::empty())
        };
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
    fn p4s<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        self.ensure_metadata()?;
        let mapping = PyDict::new(py);
        for (name, vec4) in self.event.p4s() {
            mapping.set_item(name, PyVec4(vec4))?;
        }
        Ok(mapping.into())
    }
    /// The auxiliary scalar values associated with the event
    ///
    #[getter]
    #[pyo3(name = "aux")]
    fn aux_mapping<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        self.ensure_metadata()?;
        let mapping = PyDict::new(py);
        for (name, value) in self.event.aux() {
            mapping.set_item(name, value)?;
        }
        Ok(mapping.into())
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
        Ok(self.event.p4(name).map(PyVec4))
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
        let mut resolved = Vec::new();
        for name in names {
            let selection = metadata
                .p4_selection(name)
                .ok_or_else(|| PyKeyError::new_err(format!("Unknown particle name '{name}'")))?;
            resolved.extend_from_slice(selection.indices());
        }
        Ok(resolved)
    }

    pub(crate) fn metadata_opt(&self) -> Option<&DatasetMetadata> {
        self.has_metadata.then(|| self.event.metadata())
    }
}

/// A set of Events
///
/// Datasets can be created from lists of Events or by using the constructor helpers
/// such as :meth:`laddu.Dataset.from_parquet`, :meth:`laddu.Dataset.from_root`, and
/// :meth:`laddu.Dataset.from_amptools`
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
/// aliases : dict of {str: str or list[str]}, optional
///     Additional particle identifiers that override aliases stored on the Events.
///
/// Notes
/// -----
/// Explicit metadata provided here takes precedence over metadata embedded in the
/// input Events.
///
#[pyclass(name = "Dataset", module = "laddu", subclass)]
#[derive(Clone)]
pub struct PyDataset(pub Arc<Dataset>);

#[pyclass(name = "DatasetIter", module = "laddu")]
struct PyDatasetIter {
    dataset: Arc<Dataset>,
    index: usize,
    total: usize,
}

#[pymethods]
impl PyDatasetIter {
    fn __iter__(slf: PyRef<'_, Self>) -> Py<PyDatasetIter> {
        slf.into()
    }

    fn __next__(&mut self) -> Option<PyEvent> {
        if self.index >= self.total {
            return None;
        }
        let event = self.dataset[self.index].clone();
        self.index += 1;
        Some(PyEvent {
            event,
            has_metadata: true,
        })
    }
}

#[pymethods]
impl PyDataset {
    #[new]
    #[pyo3(signature = (events, *, p4_names=None, aux_names=None, aliases=None))]
    fn new(
        events: Vec<PyEvent>,
        p4_names: Option<Vec<String>>,
        aux_names: Option<Vec<String>>,
        aliases: Option<Bound<PyDict>>,
    ) -> PyResult<Self> {
        let inferred_metadata = events
            .iter()
            .find_map(|event| event.has_metadata.then(|| event.event.metadata_arc()));

        let aliases = parse_aliases(aliases)?;
        let use_explicit_metadata =
            p4_names.is_some() || aux_names.is_some() || !aliases.is_empty();

        let metadata =
            if use_explicit_metadata {
                let resolved_p4_names = match (p4_names, inferred_metadata.as_ref()) {
                    (Some(names), _) => names,
                    (None, Some(metadata)) => metadata.p4_names().to_vec(),
                    (None, None) => Vec::new(),
                };
                let resolved_aux_names = match (aux_names, inferred_metadata.as_ref()) {
                    (Some(names), _) => names,
                    (None, Some(metadata)) => metadata.aux_names().to_vec(),
                    (None, None) => Vec::new(),
                };

                if !aliases.is_empty() && resolved_p4_names.is_empty() {
                    return Err(PyValueError::new_err(
                        "`aliases` requires `p4_names` or events with metadata for resolution",
                    ));
                }

                let mut metadata = DatasetMetadata::new(resolved_p4_names, resolved_aux_names)
                    .map_err(PyErr::from)?;
                if !aliases.is_empty() {
                    metadata
                        .add_p4_aliases(aliases.into_iter().map(|(alias_name, selection)| {
                            (alias_name, selection.into_selection())
                        }))
                        .map_err(PyErr::from)?;
                }
                Some(Arc::new(metadata))
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

    /// Read a Dataset from a Parquet file.
    ///
    /// Parameters
    /// ----------
    /// path : str or Path
    ///     The path to the Parquet file.
    /// p4s : list[str], optional
    ///     Particle identifiers corresponding to ``*_px``, ``*_py``, ``*_pz``, ``*_e`` columns.
    /// aux : list[str], optional
    ///     Auxiliary scalar column names copied verbatim in order.
    ///
    /// Notes
    /// -----
    /// If `p4s` or `aux` are not provided, they will be inferred from the column names. If all of
    /// the valid suffixes are provided for a particle, the corresponding columns will be read as a
    /// four-momentum, otherwise they will be read as auxiliary scalars.
    ///
    #[staticmethod]
    #[pyo3(signature = (path, *, p4s=None, aux=None, aliases=None))]
    fn from_parquet(
        path: Bound<PyAny>,
        p4s: Option<Vec<String>>,
        aux: Option<Vec<String>>,
        aliases: Option<Bound<PyDict>>,
    ) -> PyResult<Self> {
        let path_str = parse_dataset_path(path)?;

        let mut read_options = DatasetReadOptions::default();
        if let Some(p4s) = p4s {
            read_options = read_options.p4_names(p4s);
        }
        if let Some(aux) = aux {
            read_options = read_options.aux_names(aux);
        }
        for (alias_name, selection) in parse_aliases(aliases)?.into_iter() {
            read_options = read_options.alias(alias_name, selection);
        }
        let dataset = Dataset::from_parquet(&path_str, &read_options)?;

        Ok(Self(dataset))
    }

    /// Read a Dataset from a ROOT TTree using the oxyroot backend.
    ///
    /// Parameters
    /// ----------
    /// path : str or Path
    ///     The path to the ROOT file.
    /// tree : str, optional
    ///     Name of the TTree to read when opening ROOT files.
    /// p4s : list[str], optional
    ///     Particle identifiers corresponding to ``*_px``, ``*_py``, ``*_pz``, ``*_e`` columns.
    /// aux : list[str], optional
    ///     Auxiliary scalar column names copied verbatim in order.
    ///
    /// Notes
    /// -----
    /// If `p4s` or `aux` are not provided, they will be inferred from the column names. If all of
    /// the valid suffixes are provided for a particle, the corresponding columns will be read as a
    /// four-momentum, otherwise they will be read as auxiliary scalars.
    ///
    #[staticmethod]
    #[pyo3(signature = (path, *, tree=None, p4s=None, aux=None, aliases=None))]
    fn from_root(
        path: Bound<PyAny>,
        tree: Option<String>,
        p4s: Option<Vec<String>>,
        aux: Option<Vec<String>>,
        aliases: Option<Bound<PyDict>>,
    ) -> PyResult<Self> {
        let path_str = parse_dataset_path(path)?;

        let mut read_options = DatasetReadOptions::default();
        if let Some(p4s) = p4s {
            read_options = read_options.p4_names(p4s);
        }
        if let Some(aux) = aux {
            read_options = read_options.aux_names(aux);
        }
        if let Some(tree) = tree {
            read_options = read_options.tree(tree);
        }
        for (alias_name, selection) in parse_aliases(aliases)?.into_iter() {
            read_options = read_options.alias(alias_name, selection);
        }
        let dataset = Dataset::from_root(&path_str, &read_options)?;

        Ok(Self(dataset))
    }
    fn __len__(&self) -> usize {
        self.0.n_events()
    }
    fn __iter__(&self) -> PyDatasetIter {
        PyDatasetIter {
            dataset: self.0.clone(),
            index: 0,
            total: self.0.n_events(),
        }
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

    /// Write the dataset to a Parquet file.
    #[pyo3(signature = (path, *, chunk_size=None, precision="f64"))]
    fn to_parquet(
        &self,
        path: Bound<'_, PyAny>,
        chunk_size: Option<usize>,
        precision: &str,
    ) -> PyResult<()> {
        let path_str = parse_dataset_path(path)?;
        let mut write_options = DatasetWriteOptions::default();
        if let Some(size) = chunk_size {
            write_options.batch_size = size.max(1);
        }
        write_options.precision = parse_precision_arg(Some(precision))?;

        self.0
            .to_parquet(&path_str, &write_options)
            .map_err(PyErr::from)
    }

    /// Write the dataset to a ROOT file using the oxyroot backend.
    #[pyo3(signature = (path, *, tree=None, chunk_size=None, precision="f64"))]
    fn to_root(
        &self,
        path: Bound<'_, PyAny>,
        tree: Option<String>,
        chunk_size: Option<usize>,
        precision: &str,
    ) -> PyResult<()> {
        let path_str = parse_dataset_path(path)?;
        let mut write_options = DatasetWriteOptions::default();
        if let Some(name) = tree {
            write_options.tree = Some(name);
        }
        if let Some(size) = chunk_size {
            write_options.batch_size = size.max(1);
        }
        write_options.precision = parse_precision_arg(Some(precision))?;

        self.0
            .to_root(&path_str, &write_options)
            .map_err(PyErr::from)
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
    /// Notes
    /// -----
    /// When MPI is enabled, this returns only the events local to the current rank.
    /// Use Python iteration (`for event in dataset`, `list(dataset)`, etc.) to
    /// traverse all events across ranks.
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
    /// Examples
    /// --------
    /// >>> from laddu.utils.variables import Mass  # doctest: +SKIP
    /// >>> binned = dataset.bin_by(Mass(['kshort1']), bins=10, range=(0.9, 1.5))  # doctest: +SKIP
    /// >>> len(binned)  # doctest: +SKIP
    /// 10
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
    /// Examples
    /// --------
    /// >>> from laddu.utils.variables import Mass  # doctest: +SKIP
    /// >>> heavy = dataset.filter(Mass(['kshort1']) > 1.0)  # doctest: +SKIP
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
    /// Examples
    /// --------
    /// >>> replica = dataset.bootstrap(2024)  # doctest: +SKIP
    /// >>> len(replica) == len(dataset)  # doctest: +SKIP
    /// True
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
    /// Examples
    /// --------
    /// >>> dataset.boost_to_rest_frame_of(['kshort1', 'kshort2'])  # doctest: +SKIP
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
    /// Examples
    /// --------
    /// >>> from laddu.utils.variables import Mass  # doctest: +SKIP
    /// >>> masses = dataset.evaluate(Mass(['kshort1']))  # doctest: +SKIP
    /// >>> masses.shape  # doctest: +SKIP
    /// (len(dataset),)
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
