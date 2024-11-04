use ganesh::Observer;
use pyo3::{
    prelude::*,
    types::{PyTuple, PyTupleMethods},
};

use crate::Float;

#[pymodule]
#[allow(non_snake_case, clippy::upper_case_acronyms)]
pub(crate) mod laddu {
    use std::array;
    use std::sync::Arc;

    use super::*;
    use crate as rust;
    use crate::likelihoods::LikelihoodTerm as RustLikelihoodTerm;
    use crate::likelihoods::MinimizerOptions;
    use crate::utils::variables::Variable;
    use crate::utils::vectors::{FourMomentum, FourVector, ThreeMomentum, ThreeVector};
    use crate::Float;
    use ganesh::algorithms::lbfgsb::{LBFGSBFTerminator, LBFGSBGTerminator};
    use ganesh::algorithms::nelder_mead::{
        NelderMeadFTerminator, NelderMeadXTerminator, SimplexExpansionMethod,
    };
    use ganesh::algorithms::{NelderMead, LBFGSB};
    use num::Complex;
    use numpy::{PyArray1, PyArray2};
    use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
    use pyo3::types::{PyDict, PyList};

    #[pyfunction]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// A 3-momentum vector formed from Cartesian components
    ///
    /// Parameters
    /// ----------
    /// px, py, pz : float
    ///     The Cartesian components of the 3-vector
    ///
    /// Returns
    /// -------
    /// Vector3
    ///     A new 3-momentum vector made from the given components
    ///
    #[pyclass]
    #[derive(Clone)]
    struct Vector3(nalgebra::Vector3<Float>);
    #[pymethods]
    impl Vector3 {
        #[new]
        fn new(px: Float, py: Float, pz: Float) -> Self {
            Self(nalgebra::Vector3::new(px, py, pz))
        }
        fn __add__(&self, other: Self) -> Self {
            Self(self.0 + other.0)
        }
        /// The dot product
        ///
        /// Calculates the dot product of two Vector3s.
        ///
        /// Parameters
        /// ----------
        /// other : Vector3
        ///     A vector input with which the dot product is taken
        ///
        /// Returns
        /// -------
        /// float
        ///     The dot product of this vector and `other`
        ///
        pub fn dot(&self, other: Self) -> Float {
            self.0.dot(&other.0)
        }
        /// The cross product
        ///
        /// Calculates the cross product of two Vector3s.
        ///
        /// Parameters
        /// ----------
        /// other : Vector3
        ///     A vector input with which the cross product is taken
        ///
        /// Returns
        /// -------
        /// Vector3
        ///     The cross product of this vector and `other`
        ///
        fn cross(&self, other: Self) -> Self {
            Self(self.0.cross(&other.0))
        }
        /// The magnitude of the 3-vector
        ///
        /// This is calculated as:
        ///
        /// .. math:: \sqrt{p_x^2 + p_y^2 + p_z^2}
        ///
        /// Returns
        /// -------
        /// float
        ///     The magnitude of this vector
        #[getter]
        fn mag(&self) -> Float {
            self.0.mag()
        }
        /// The magnitude-squared of the 3-vector
        ///
        /// This is calculated as:
        ///
        /// .. math:: p_x^2 + p_y^2 + p_z^2
        ///
        /// Returns
        /// -------
        /// float
        ///     The squared magnitude of this vector
        ///
        #[getter]
        fn mag2(&self) -> Float {
            self.0.mag2()
        }
        /// The cosine of the polar angle of this vector in spherical coordinates
        ///
        /// The polar angle is defined in the range
        ///
        /// .. math:: 0 \leq \theta \leq \pi
        ///
        /// so the cosine falls in the range
        ///
        /// .. math:: -1 \leq \cos\theta \leq +1
        ///
        /// This is calculated as:
        ///
        /// .. math:: \cos\theta = \frac{p_z}{|\vec{p}|}
        ///
        /// Returns
        /// -------
        /// float
        ///     The cosine of the polar angle of this vector
        ///
        #[getter]
        fn costheta(&self) -> Float {
            self.0.costheta()
        }
        /// The polar angle of this vector in spherical coordinates
        ///
        /// The polar angle is defined in the range
        ///
        /// .. math:: 0 \leq \theta \leq \pi
        ///
        /// This is calculated as:
        ///
        /// .. math:: \theta = \arccos\left(\frac{p_z}{|\vec{p}|}\right)
        ///
        /// Returns
        /// -------
        /// float
        ///     The polar angle of this vector
        ///
        #[getter]
        fn theta(&self) -> Float {
            self.0.theta()
        }
        /// The azimuthal angle of this vector in spherical coordinates
        ///
        /// The azimuthal angle is defined in the range
        ///
        /// .. math:: 0 \leq \varphi \leq 2\pi
        ///
        /// This is calculated as:
        ///
        /// .. math:: \varphi = \text{sgn}(p_y)\arccos\left(\frac{p_x}{\sqrt{p_x^2 + p_y^2}}\right)
        ///
        /// although the actual calculation just uses the `atan2` function
        ///
        /// Returns
        /// -------
        /// float
        ///     The azimuthal angle of this vector
        ///
        #[getter]
        fn phi(&self) -> Float {
            self.0.phi()
        }
        /// The normalized unit vector pointing in the direction of this vector
        ///
        /// Returns
        /// -------
        /// Vector3
        ///     A unit vector pointing in the same direction as this vector
        ///
        #[getter]
        fn unit(&self) -> Self {
            Self(self.0.unit())
        }
        /// The x-component of this vector
        ///
        /// Returns
        /// -------
        /// float
        ///     The x-component
        ///
        #[getter]
        fn px(&self) -> Float {
            self.0.px()
        }
        /// The y-component of this vector
        ///
        /// Returns
        /// -------
        /// float
        ///     The y-component
        ///
        #[getter]
        fn py(&self) -> Float {
            self.0.py()
        }
        /// The z-component of this vector
        ///
        /// Returns
        /// -------
        /// float
        ///     The z-component
        ///
        #[getter]
        fn pz(&self) -> Float {
            self.0.pz()
        }
        /// Convert the 3-vector to a `numpy` array
        ///
        /// Returns
        /// -------
        /// numpy_vec: array_like
        ///     A `numpy` array built from the components of this `Vector3`
        ///
        fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, self.0.as_slice())
        }
        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
        fn __str__(&self) -> String {
            format!("{}", self.0)
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct Vector4(nalgebra::Vector4<Float>);
    #[pymethods]
    impl Vector4 {
        #[new]
        fn new(e: Float, px: Float, py: Float, pz: Float) -> Self {
            Self(nalgebra::Vector4::new(e, px, py, pz))
        }
        fn __add__(&self, other: Self) -> Self {
            Self(self.0 + other.0)
        }
        #[getter]
        fn mag(&self) -> Float {
            self.0.mag()
        }
        #[getter]
        fn mag2(&self) -> Float {
            self.0.mag2()
        }
        #[getter]
        fn vec3(&self) -> Vector3 {
            Vector3(self.0.vec3().into())
        }
        fn boost(&self, beta: &Vector3) -> Self {
            Self(self.0.boost(&beta.0))
        }
        #[getter]
        fn e(&self) -> Float {
            self.0[0]
        }
        #[getter]
        fn px(&self) -> Float {
            self.0.px()
        }
        #[getter]
        fn py(&self) -> Float {
            self.0.py()
        }
        #[getter]
        fn pz(&self) -> Float {
            self.0.pz()
        }
        #[getter]
        fn momentum(&self) -> Vector3 {
            Vector3(self.0.momentum().into())
        }
        #[getter]
        fn gamma(&self) -> Float {
            self.0.gamma()
        }
        #[getter]
        fn beta(&self) -> Vector3 {
            Vector3(self.0.beta())
        }
        #[getter]
        fn m(&self) -> Float {
            self.0.m()
        }
        #[getter]
        fn m2(&self) -> Float {
            self.0.m2()
        }
        fn boost_along(&self, other: &Self) -> Self {
            Self(self.0.boost_along(&other.0))
        }
        #[staticmethod]
        fn from_momentum(momentum: &Vector3, mass: Float) -> Self {
            Self(nalgebra::Vector4::from_momentum(&momentum.0, mass))
        }
        fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, self.0.as_slice())
        }
        fn __str__(&self) -> String {
            format!("{}", self.0)
        }
        fn __repr__(&self) -> String {
            self.0.to_p4_string()
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct Event(Arc<rust::data::Event>);

    #[pymethods]
    impl Event {
        #[new]
        pub(crate) fn new(p4s: Vec<Vector4>, eps: Vec<Vector3>, weight: Float) -> Self {
            Self(Arc::new(rust::data::Event {
                p4s: p4s.into_iter().map(|arr| arr.0).collect(),
                eps: eps.into_iter().map(|arr| arr.0).collect(),
                weight,
            }))
        }
        pub(crate) fn __str__(&self) -> String {
            self.0.to_string()
        }
        #[getter]
        pub(crate) fn get_p4s(&self) -> Vec<Vector4> {
            self.0.p4s.iter().map(|p4| Vector4(*p4)).collect()
        }
        #[getter]
        pub(crate) fn get_eps(&self) -> Vec<Vector3> {
            self.0.eps.iter().map(|eps_vec| Vector3(*eps_vec)).collect()
        }
        #[getter]
        pub(crate) fn get_weight(&self) -> Float {
            self.0.weight
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct Dataset(Arc<rust::data::Dataset>);

    #[pymethods]
    impl Dataset {
        #[new]
        fn new(events: Vec<Event>) -> Self {
            Self(Arc::new(rust::data::Dataset {
                events: events.into_iter().map(|event| event.0).collect(),
            }))
        }
        fn __len__(&self) -> usize {
            self.0.len()
        }
        fn len(&self) -> usize {
            self.0.len()
        }
        fn weighted_len(&self) -> Float {
            self.0.weighted_len()
        }
        #[getter]
        fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, &self.0.weights())
        }
        #[getter]
        fn events(&self) -> Vec<Event> {
            self.0
                .events
                .iter()
                .map(|rust_event| Event(rust_event.clone()))
                .collect()
        }
        fn __getitem__(&self, index: usize) -> PyResult<Event> {
            self.0
                .get(index)
                .ok_or(PyIndexError::new_err("index out of range"))
                .map(|rust_event| Event(rust_event.clone()))
        }
        #[pyo3(signature = (variable, bins, range))]
        fn bin_by(
            &self,
            variable: Bound<'_, PyAny>,
            bins: usize,
            range: (Float, Float),
        ) -> PyResult<BinnedDataset> {
            let rust_variable = if let Ok(py_mass) = variable.extract::<PyRef<Mass>>() {
                py_mass.0.clone()
            } else {
                return Err(PyTypeError::new_err("Unsupported variable!"));
            };
            Ok(BinnedDataset(self.0.bin_by(rust_variable, bins, range)))
        }
        fn bootstrap(&self, seed: usize) -> Dataset {
            Dataset(self.0.bootstrap(seed))
        }
    }

    #[pyclass]
    struct BinnedDataset(rust::data::BinnedDataset);

    #[pymethods]
    impl BinnedDataset {
        fn __len__(&self) -> usize {
            self.0.len()
        }
        fn len(&self) -> usize {
            self.0.len()
        }
        #[getter]
        fn bins(&self) -> usize {
            self.0.bins()
        }
        #[getter]
        fn range(&self) -> (Float, Float) {
            self.0.range()
        }
        #[getter]
        fn edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, &self.0.edges())
        }
        fn __getitem__(&self, index: usize) -> PyResult<Dataset> {
            self.0
                .get(index)
                .ok_or(PyIndexError::new_err("index out of range"))
                .map(|rust_dataset| Dataset(rust_dataset.clone()))
        }
    }

    #[pyfunction]
    fn open(path: &str) -> PyResult<Dataset> {
        Ok(Dataset(rust::data::open(path)?))
    }

    #[pyclass]
    #[derive(Clone)]
    struct Mass(rust::utils::variables::Mass);

    #[pymethods]
    impl Mass {
        #[new]
        fn new(constituents: Vec<usize>) -> Self {
            Self(rust::utils::variables::Mass::new(&constituents))
        }
        fn value(&self, event: &Event) -> Float {
            self.0.value(&event.0)
        }
        fn value_on<'py>(&self, py: Python<'py>, dataset: &Dataset) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, &self.0.value_on(&dataset.0))
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct CosTheta(rust::utils::variables::CosTheta);

    #[pymethods]
    impl CosTheta {
        #[new]
        #[pyo3(signature=(beam, recoil, daughter, resonance, frame="Helicity"))]
        fn new(
            beam: usize,
            recoil: Vec<usize>,
            daughter: Vec<usize>,
            resonance: Vec<usize>,
            frame: &str,
        ) -> Self {
            Self(rust::utils::variables::CosTheta::new(
                beam,
                &recoil,
                &daughter,
                &resonance,
                frame.parse().unwrap(),
            ))
        }
        fn value(&self, event: &Event) -> Float {
            self.0.value(&event.0)
        }
        fn value_on<'py>(&self, py: Python<'py>, dataset: &Dataset) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, &self.0.value_on(&dataset.0))
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct Phi(rust::utils::variables::Phi);

    #[pymethods]
    impl Phi {
        #[new]
        #[pyo3(signature=(beam, recoil, daughter, resonance, frame="Helicity"))]
        fn new(
            beam: usize,
            recoil: Vec<usize>,
            daughter: Vec<usize>,
            resonance: Vec<usize>,
            frame: &str,
        ) -> Self {
            Self(rust::utils::variables::Phi::new(
                beam,
                &recoil,
                &daughter,
                &resonance,
                frame.parse().unwrap(),
            ))
        }
        fn value(&self, event: &Event) -> Float {
            self.0.value(&event.0)
        }
        fn value_on<'py>(&self, py: Python<'py>, dataset: &Dataset) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, &self.0.value_on(&dataset.0))
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct Angles(rust::utils::variables::Angles);
    #[pymethods]
    impl Angles {
        #[new]
        #[pyo3(signature=(beam, recoil, daughter, resonance, frame="Helicity"))]
        fn new(
            beam: usize,
            recoil: Vec<usize>,
            daughter: Vec<usize>,
            resonance: Vec<usize>,
            frame: &str,
        ) -> Self {
            Self(rust::utils::variables::Angles::new(
                beam,
                &recoil,
                &daughter,
                &resonance,
                frame.parse().unwrap(),
            ))
        }
        #[getter]
        fn costheta(&self) -> CosTheta {
            CosTheta(self.0.costheta.clone())
        }
        #[getter]
        fn phi(&self) -> Phi {
            Phi(self.0.phi.clone())
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct PolAngle(rust::utils::variables::PolAngle);

    #[pymethods]
    impl PolAngle {
        #[new]
        fn new(beam: usize, recoil: Vec<usize>) -> Self {
            Self(rust::utils::variables::PolAngle::new(beam, &recoil))
        }
        fn value(&self, event: &Event) -> Float {
            self.0.value(&event.0)
        }
        fn value_on<'py>(&self, py: Python<'py>, dataset: &Dataset) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, &self.0.value_on(&dataset.0))
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct PolMagnitude(rust::utils::variables::PolMagnitude);

    #[pymethods]
    impl PolMagnitude {
        #[new]
        fn new(beam: usize) -> Self {
            Self(rust::utils::variables::PolMagnitude::new(beam))
        }
        fn value(&self, event: &Event) -> Float {
            self.0.value(&event.0)
        }
        fn value_on<'py>(&self, py: Python<'py>, dataset: &Dataset) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, &self.0.value_on(&dataset.0))
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct Polarization(rust::utils::variables::Polarization);
    #[pymethods]
    impl Polarization {
        #[new]
        fn new(beam: usize, recoil: Vec<usize>) -> Self {
            Polarization(rust::utils::variables::Polarization::new(beam, &recoil))
        }
        #[getter]
        fn pol_magnitude(&self) -> PolMagnitude {
            PolMagnitude(self.0.pol_magnitude)
        }
        #[getter]
        fn pol_angle(&self) -> PolAngle {
            PolAngle(self.0.pol_angle.clone())
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct AmplitudeID(rust::amplitudes::AmplitudeID);

    #[pyclass]
    #[derive(Clone)]
    pub(crate) struct Expression(pub(crate) rust::amplitudes::Expression);

    #[pymethods]
    impl AmplitudeID {
        fn real(&self) -> Expression {
            Expression(self.0.real())
        }
        fn imag(&self) -> Expression {
            Expression(self.0.imag())
        }
        fn norm_sqr(&self) -> Expression {
            Expression(self.0.norm_sqr())
        }
        fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(self.0.clone() + other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(self.0.clone() + other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(self.0.clone() * other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(self.0.clone() * other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for *"))
            }
        }
        fn __str__(&self) -> String {
            format!("{}", self.0)
        }
        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
    }

    #[pymethods]
    impl Expression {
        fn real(&self) -> Expression {
            Expression(self.0.real())
        }
        fn imag(&self) -> Expression {
            Expression(self.0.imag())
        }
        fn norm_sqr(&self) -> Expression {
            Expression(self.0.norm_sqr())
        }
        fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(self.0.clone() + other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(self.0.clone() + other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(self.0.clone() * other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(self.0.clone() * other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for *"))
            }
        }
        fn __str__(&self) -> String {
            format!("{}", self.0)
        }
        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
    }

    #[pyclass]
    struct Manager(rust::amplitudes::Manager);

    #[pyclass]
    struct Amplitude(Box<dyn rust::amplitudes::Amplitude>);

    #[pymethods]
    impl Manager {
        #[new]
        fn new() -> Self {
            Self(rust::amplitudes::Manager::default())
        }
        fn register(&mut self, amplitude: &Amplitude) -> PyResult<AmplitudeID> {
            Ok(AmplitudeID(self.0.register(amplitude.0.clone())?))
        }
        fn load(&self, dataset: &Dataset, expression: &Expression) -> Evaluator {
            Evaluator(self.0.load(&dataset.0, &expression.0))
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct Evaluator(rust::amplitudes::Evaluator);

    #[pymethods]
    impl Evaluator {
        #[getter]
        fn parameters(&self) -> Vec<String> {
            self.0.parameters()
        }
        fn activate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.activate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.activate_many(&vec);
            } else {
                return Err(PyTypeError::new_err(
                    "Argument must be either a string or a list of strings",
                ));
            }
            Ok(())
        }
        fn activate_all(&self) {
            self.0.activate_all();
        }
        fn deactivate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.deactivate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.deactivate_many(&vec);
            } else {
                return Err(PyTypeError::new_err(
                    "Argument must be either a string or a list of strings",
                ));
            }
            Ok(())
        }
        fn deactivate_all(&self) {
            self.0.deactivate_all();
        }
        fn isolate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.isolate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.isolate_many(&vec);
            } else {
                return Err(PyTypeError::new_err(
                    "Argument must be either a string or a list of strings",
                ));
            }
            Ok(())
        }
        fn evaluate<'py>(
            &self,
            py: Python<'py>,
            parameters: Vec<Float>,
        ) -> Bound<'py, PyArray1<Complex<Float>>> {
            PyArray1::from_slice_bound(py, &self.0.evaluate(&parameters))
        }
    }

    trait GetStrExtractObj {
        fn get_extract<T>(&self, key: &str) -> PyResult<Option<T>>
        where
            T: for<'py> FromPyObject<'py>;
    }

    impl GetStrExtractObj for Bound<'_, PyDict> {
        fn get_extract<T>(&self, key: &str) -> PyResult<Option<T>>
        where
            T: for<'py> FromPyObject<'py>,
        {
            self.get_item(key)?
                .map(|value| value.extract::<T>())
                .transpose()
        }
    }

    fn _parse_minimizer_options(
        n_parameters: usize,
        method: &str,
        max_steps: usize,
        debug: bool,
        verbose: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<MinimizerOptions> {
        let mut options = MinimizerOptions::default();
        let mut show_step = true;
        let mut show_x = true;
        let mut show_fx = true;
        if let Some(kwargs) = kwargs {
            show_step = kwargs.get_extract::<bool>("show_step")?.unwrap_or(true);
            show_x = kwargs.get_extract::<bool>("show_x")?.unwrap_or(true);
            show_fx = kwargs.get_extract::<bool>("show_fx")?.unwrap_or(true);
            let tol_x_rel = kwargs
                .get_extract::<Float>("tol_x_rel")?
                .unwrap_or(Float::EPSILON);
            let tol_x_abs = kwargs
                .get_extract::<Float>("tol_x_abs")?
                .unwrap_or(Float::EPSILON);
            let tol_f_rel = kwargs
                .get_extract::<Float>("tol_f_rel")?
                .unwrap_or(Float::EPSILON);
            let tol_f_abs = kwargs
                .get_extract::<Float>("tol_f_abs")?
                .unwrap_or(Float::EPSILON);
            let tol_g_abs = kwargs
                .get_extract::<Float>("tol_g_abs")?
                .unwrap_or(Float::cbrt(Float::EPSILON));
            let g_tolerance = kwargs.get_extract::<Float>("g_tolerance")?.unwrap_or(1e-5);
            let adaptive = kwargs.get_extract::<bool>("adaptive")?.unwrap_or(false);
            let alpha = kwargs.get_extract::<Float>("alpha")?;
            let beta = kwargs.get_extract::<Float>("beta")?;
            let gamma = kwargs.get_extract::<Float>("gamma")?;
            let delta = kwargs.get_extract::<Float>("delta")?;
            let simplex_expansion_method = kwargs
                .get_extract::<String>("simplex_expansion_method")?
                .unwrap_or("greedy minimization".into());
            let nelder_mead_f_terminator = kwargs
                .get_extract::<String>("nelder_mead_f_terminator")?
                .unwrap_or("stddev".into());
            let nelder_mead_x_terminator = kwargs
                .get_extract::<String>("nelder_mead_x_terminator")?
                .unwrap_or("singer".into());
            let mut observers: Vec<PyObserver> = Vec::default();
            // } else if let Ok(list_arg) = arg.downcast::<PyList>() {
            //     let vec: Vec<String> = list_arg.extract()?;
            if let Ok(Some(observer_arg)) = kwargs.get_item("observers") {
                if let Ok(observer_list) = observer_arg.downcast::<PyList>() {
                    for item in observer_list.iter() {
                        let observer = item.extract::<PyObserver>()?;
                        observers.push(observer);
                    }
                } else if let Ok(single_observer) = observer_arg.extract::<PyObserver>() {
                    observers.push(single_observer);
                } else {
                    return Err(PyTypeError::new_err("The keyword argument \"observers\" must either be a single Observer or a list of Observers!"));
                }
            }
            for observer in observers {
                options = options.with_observer(observer);
            }
            match method {
                "lbfgsb" => {
                    options = options.with_algorithm(
                        LBFGSB::default()
                            .with_terminator_f(LBFGSBFTerminator { tol_f_abs })
                            .with_terminator_g(LBFGSBGTerminator { tol_g_abs })
                            .with_g_tolerance(g_tolerance),
                    )
                }
                "nelder_mead" => {
                    let terminator_f = match nelder_mead_f_terminator.as_str() {
                        "amoeba" => NelderMeadFTerminator::Amoeba { tol_f_rel },
                        "absolute" => NelderMeadFTerminator::Absolute { tol_f_abs },
                        "stddev" => NelderMeadFTerminator::StdDev { tol_f_abs },
                        "none" => NelderMeadFTerminator::None,
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "Invalid \"nelder_mead_f_terminator\": \"{}\"",
                                nelder_mead_f_terminator
                            )))
                        }
                    };
                    let terminator_x = match nelder_mead_x_terminator.as_str() {
                        "diameter" => NelderMeadXTerminator::Diameter { tol_x_abs },
                        "higham" => NelderMeadXTerminator::Higham { tol_x_rel },
                        "rowan" => NelderMeadXTerminator::Rowan { tol_x_rel },
                        "singer" => NelderMeadXTerminator::Singer { tol_x_rel },
                        "none" => NelderMeadXTerminator::None,
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "Invalid \"nelder_mead_x_terminator\": \"{}\"",
                                nelder_mead_x_terminator
                            )))
                        }
                    };
                    let simplex_expansion_method = match simplex_expansion_method.as_str() {
                        "greedy minimization" => SimplexExpansionMethod::GreedyMinimization,
                        "greedy expansion" => SimplexExpansionMethod::GreedyExpansion,
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "Invalid \"simplex_expansion_method\": \"{}\"",
                                simplex_expansion_method
                            )))
                        }
                    };
                    let mut nelder_mead = NelderMead::default()
                        .with_terminator_f(terminator_f)
                        .with_terminator_x(terminator_x)
                        .with_expansion_method(simplex_expansion_method);
                    if adaptive {
                        nelder_mead = nelder_mead.with_adaptive(n_parameters);
                    }
                    if let Some(alpha) = alpha {
                        nelder_mead = nelder_mead.with_alpha(alpha);
                    }
                    if let Some(beta) = beta {
                        nelder_mead = nelder_mead.with_beta(beta);
                    }
                    if let Some(gamma) = gamma {
                        nelder_mead = nelder_mead.with_gamma(gamma);
                    }
                    if let Some(delta) = delta {
                        nelder_mead = nelder_mead.with_delta(delta);
                    }
                    options = options.with_algorithm(nelder_mead)
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid \"method\": \"{}\"",
                        method
                    )))
                }
            }
        }
        if debug {
            options = options.debug();
        }
        if verbose {
            options = options.verbose(show_step, show_x, show_fx);
        }
        options = options.with_max_steps(max_steps);
        Ok(options)
    }

    #[pyclass]
    #[derive(Clone)]
    struct NLL(Box<rust::likelihoods::NLL>);

    #[pymethods]
    impl NLL {
        #[new]
        fn new(
            manager: &Manager,
            ds_data: &Dataset,
            ds_mc: &Dataset,
            expression: &Expression,
        ) -> Self {
            Self(rust::likelihoods::NLL::new(
                &manager.0,
                &ds_data.0,
                &ds_mc.0,
                &expression.0,
            ))
        }
        #[getter]
        fn data(&self) -> Dataset {
            Dataset(self.0.data_evaluator.dataset.clone())
        }
        #[getter]
        fn mc(&self) -> Dataset {
            Dataset(self.0.mc_evaluator.dataset.clone())
        }
        fn as_term(&self) -> LikelihoodTerm {
            LikelihoodTerm(self.0.clone())
        }
        #[getter]
        fn parameters(&self) -> Vec<String> {
            self.0.parameters()
        }
        fn activate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.activate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.activate_many(&vec);
            } else {
                return Err(PyTypeError::new_err(
                    "Argument must be either a string or a list of strings",
                ));
            }
            Ok(())
        }
        fn activate_all(&self) {
            self.0.activate_all();
        }
        fn deactivate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.deactivate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.deactivate_many(&vec);
            } else {
                return Err(PyTypeError::new_err(
                    "Argument must be either a string or a list of strings",
                ));
            }
            Ok(())
        }
        fn deactivate_all(&self) {
            self.0.deactivate_all();
        }
        fn isolate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.isolate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.isolate_many(&vec);
            } else {
                return Err(PyTypeError::new_err(
                    "Argument must be either a string or a list of strings",
                ));
            }
            Ok(())
        }
        fn evaluate(&self, parameters: Vec<Float>) -> Float {
            self.0.evaluate(&parameters)
        }
        fn project<'py>(
            &self,
            py: Python<'py>,
            parameters: Vec<Float>,
        ) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, &self.0.project(&parameters))
        }
        #[pyo3(signature = (p0, bounds=None, method="lbfgsb", max_steps=4000, debug=false, verbose=false, **kwargs))]
        #[allow(clippy::too_many_arguments)]
        fn minimize(
            &self,
            p0: Vec<Float>,
            bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
            method: &str,
            max_steps: usize,
            debug: bool,
            verbose: bool,
            kwargs: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Status> {
            let bounds = bounds.map(|bounds_vec| {
                bounds_vec
                    .iter()
                    .map(|(opt_lb, opt_ub)| {
                        (
                            opt_lb.unwrap_or(Float::NEG_INFINITY),
                            opt_ub.unwrap_or(Float::INFINITY),
                        )
                    })
                    .collect()
            });
            let n_parameters = p0.len();
            let options =
                _parse_minimizer_options(n_parameters, method, max_steps, debug, verbose, kwargs)?;
            let status = self.0.minimize(&p0, bounds, Some(options));
            Ok(Status(status))
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct LikelihoodTerm(Box<dyn rust::likelihoods::LikelihoodTerm>);

    #[pyclass]
    #[derive(Clone)]
    struct LikelihoodID(rust::likelihoods::LikelihoodID);

    #[pyclass]
    #[derive(Clone)]
    struct LikelihoodExpression(rust::likelihoods::LikelihoodExpression);

    #[pymethods]
    impl LikelihoodID {
        fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(self.0.clone() + other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(self.0.clone() + other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(self.0.clone() * other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(self.0.clone() * other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for *"))
            }
        }
        fn __str__(&self) -> String {
            format!("{}", self.0)
        }
        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
    }

    #[pymethods]
    impl LikelihoodExpression {
        fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(self.0.clone() + other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(self.0.clone() + other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(self.0.clone() * other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(self.0.clone() * other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for *"))
            }
        }
        fn __str__(&self) -> String {
            format!("{}", self.0)
        }
        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct LikelihoodManager(rust::likelihoods::LikelihoodManager);

    #[pymethods]
    impl LikelihoodManager {
        #[new]
        fn new() -> Self {
            Self(rust::likelihoods::LikelihoodManager::default())
        }
        fn register(&mut self, likelihood_term: &LikelihoodTerm) -> LikelihoodID {
            LikelihoodID(self.0.register(likelihood_term.0.clone()))
        }
        fn parameters(&self) -> Vec<String> {
            self.0.parameters()
        }
        fn load(&self, likelihood_expression: &LikelihoodExpression) -> LikelihoodEvaluator {
            LikelihoodEvaluator(self.0.load(&likelihood_expression.0))
        }
    }

    #[pyclass]
    struct LikelihoodEvaluator(rust::likelihoods::LikelihoodEvaluator);

    #[pymethods]
    impl LikelihoodEvaluator {
        #[getter]
        fn parameters(&self) -> Vec<String> {
            self.0.parameters()
        }
        fn evaluate(&self, parameters: Vec<Float>) -> Float {
            self.0.evaluate(&parameters)
        }
        #[pyo3(signature = (p0, bounds=None, method="lbfgsb", max_steps=4000, debug=false, verbose=false, **kwargs))]
        #[allow(clippy::too_many_arguments)]
        fn minimize(
            &self,
            p0: Vec<Float>,
            bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
            method: &str,
            max_steps: usize,
            debug: bool,
            verbose: bool,
            kwargs: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Status> {
            let bounds = bounds.map(|bounds_vec| {
                bounds_vec
                    .iter()
                    .map(|(opt_lb, opt_ub)| {
                        (
                            opt_lb.unwrap_or(Float::NEG_INFINITY),
                            opt_ub.unwrap_or(Float::INFINITY),
                        )
                    })
                    .collect()
            });
            let n_parameters = p0.len();
            let options =
                _parse_minimizer_options(n_parameters, method, max_steps, debug, verbose, kwargs)?;
            let status = self.0.minimize(&p0, bounds, Some(options));
            Ok(Status(status))
        }
    }

    #[pyfunction]
    fn LikelihoodScalar(name: String) -> LikelihoodTerm {
        LikelihoodTerm(rust::likelihoods::LikelihoodScalar::new(name))
    }

    #[pyclass]
    #[pyo3(name = "Observer")]
    pub(crate) struct PyObserver(pub(crate) Py<PyAny>);

    #[pymethods]
    impl PyObserver {
        #[new]
        pub fn new(observer: Py<PyAny>) -> Self {
            Self(observer)
        }
    }

    #[pyclass]
    #[derive(Clone)]
    pub(crate) struct Status(pub(crate) ganesh::Status<Float>);
    #[pymethods]
    impl Status {
        #[getter]
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, self.0.x.as_slice())
        }
        #[getter]
        fn err<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<Float>>> {
            self.0
                .err
                .clone()
                .map(|err| PyArray1::from_slice_bound(py, err.as_slice()))
        }
        #[getter]
        fn x0<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, self.0.x0.as_slice())
        }
        #[getter]
        fn fx(&self) -> Float {
            self.0.fx
        }
        #[getter]
        fn cov<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<Float>>> {
            self.0.cov.clone().map(|cov| {
                PyArray2::from_vec2_bound(
                    py,
                    &cov.row_iter()
                        .map(|row| row.iter().cloned().collect())
                        .collect::<Vec<Vec<Float>>>(),
                )
                .unwrap()
            })
        }
        #[getter]
        fn hess<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<Float>>> {
            self.0.hess.clone().map(|hess| {
                PyArray2::from_vec2_bound(
                    py,
                    &hess
                        .row_iter()
                        .map(|row| row.iter().cloned().collect())
                        .collect::<Vec<Vec<Float>>>(),
                )
                .unwrap()
            })
        }
        #[getter]
        fn message(&self) -> String {
            self.0.message.clone()
        }
        #[getter]
        fn converged(&self) -> bool {
            self.0.converged
        }
        #[getter]
        fn bounds(&self) -> Option<Vec<ParameterBound>> {
            self.0
                .bounds
                .clone()
                .map(|bounds| bounds.iter().map(|bound| ParameterBound(*bound)).collect())
        }
        #[getter]
        fn n_f_evals(&self) -> usize {
            self.0.n_f_evals
        }
        #[getter]
        fn n_g_evals(&self) -> usize {
            self.0.n_g_evals
        }
        fn __str__(&self) -> String {
            self.0.to_string()
        }
        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
    }

    #[pyclass]
    #[derive(Clone)]
    #[pyo3(name = "Bound")]
    struct ParameterBound(ganesh::Bound<Float>);
    #[pymethods]
    impl ParameterBound {
        #[getter]
        fn lower(&self) -> Float {
            self.0.lower()
        }
        #[getter]
        fn upper(&self) -> Float {
            self.0.upper()
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct ParameterLike(rust::amplitudes::ParameterLike);

    #[pyfunction]
    fn parameter(name: &str) -> ParameterLike {
        ParameterLike(rust::amplitudes::parameter(name))
    }

    #[pyfunction]
    fn constant(value: Float) -> ParameterLike {
        ParameterLike(rust::amplitudes::constant(value))
    }

    #[pyfunction]
    fn Scalar(name: &str, value: ParameterLike) -> Amplitude {
        Amplitude(rust::amplitudes::common::Scalar::new(name, value.0))
    }

    #[pyfunction]
    fn ComplexScalar(name: &str, re: ParameterLike, im: ParameterLike) -> Amplitude {
        Amplitude(rust::amplitudes::common::ComplexScalar::new(
            name, re.0, im.0,
        ))
    }

    #[pyfunction]
    fn PolarComplexScalar(name: &str, r: ParameterLike, theta: ParameterLike) -> Amplitude {
        Amplitude(rust::amplitudes::common::PolarComplexScalar::new(
            name, r.0, theta.0,
        ))
    }

    #[pyfunction]
    fn Ylm(name: &str, l: usize, m: isize, angles: &Angles) -> Amplitude {
        Amplitude(rust::amplitudes::ylm::Ylm::new(name, l, m, &angles.0))
    }

    #[pyfunction]
    fn Zlm(
        name: &str,
        l: usize,
        m: isize,
        r: &str,
        angles: &Angles,
        polarization: &Polarization,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::zlm::Zlm::new(
            name,
            l,
            m,
            r.parse().unwrap(),
            &angles.0,
            &polarization.0,
        ))
    }

    #[pyfunction]
    fn BreitWigner(
        name: &str,
        mass: ParameterLike,
        width: ParameterLike,
        l: usize,
        daughter_1_mass: &Mass,
        daughter_2_mass: &Mass,
        resonance_mass: &Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::breit_wigner::BreitWigner::new(
            name,
            mass.0,
            width.0,
            l,
            &daughter_1_mass.0,
            &daughter_2_mass.0,
            &resonance_mass.0,
        ))
    }

    #[pyfunction]
    fn KopfKMatrixF0(
        name: &str,
        couplings: [[ParameterLike; 2]; 5],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixF0::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }

    #[pyfunction]
    fn KopfKMatrixF2(
        name: &str,
        couplings: [[ParameterLike; 2]; 4],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixF2::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }

    #[pyfunction]
    fn KopfKMatrixA0(
        name: &str,
        couplings: [[ParameterLike; 2]; 2],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixA0::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }

    #[pyfunction]
    fn KopfKMatrixA2(
        name: &str,
        couplings: [[ParameterLike; 2]; 2],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixA2::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }

    #[pyfunction]
    fn KopfKMatrixRho(
        name: &str,
        couplings: [[ParameterLike; 2]; 2],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixRho::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }

    #[pyfunction]
    fn KopfKMatrixPi1(
        name: &str,
        couplings: [[ParameterLike; 2]; 1],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixPi1::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }
}

impl Observer<Float, ()> for crate::python::laddu::PyObserver {
    fn callback(
        &mut self,
        step: usize,
        status: &mut ganesh::Status<Float>,
        _user_data: &mut (),
    ) -> bool {
        let (new_status, result) = Python::with_gil(|py| {
            let res = self
                .0
                .bind(py)
                .call_method(
                    "callback",
                    (step, crate::python::laddu::Status(status.clone())),
                    None,
                )
                .unwrap();
            let res_tuple = res.downcast::<PyTuple>().unwrap();
            let new_status = res_tuple
                .get_item(0)
                .unwrap()
                .extract::<crate::python::laddu::Status>()
                .unwrap()
                .0;
            let result = res_tuple.get_item(1).unwrap().extract::<bool>().unwrap();
            (new_status, result)
        });
        *status = new_status;
        result
    }
}
impl FromPyObject<'_> for crate::python::laddu::PyObserver {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(crate::python::laddu::PyObserver(ob.clone().into()))
    }
}
