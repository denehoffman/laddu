use pyo3::prelude::*;

#[pymodule]
#[allow(non_snake_case, clippy::upper_case_acronyms)]
mod laddu {
    use std::array;
    use std::sync::Arc;

    use super::*;
    use crate as rust;
    use crate::utils::variables::Variable;
    use crate::utils::vectors::{FourMomentum, FourVector, ThreeMomentum, ThreeVector};
    use crate::Float;
    use num::Complex;
    use numpy::PyArray1;
    use pyo3::exceptions::{PyIndexError, PyTypeError};
    use pyo3::types::PyList;

    #[pyfunction]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

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
        fn dot(&self, other: Self) -> Float {
            self.0.dot(&other.0)
        }
        fn cross(&self, other: Self) -> Self {
            Self(self.0.cross(&other.0))
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
        fn costheta(&self) -> Float {
            self.0.costheta()
        }
        #[getter]
        fn theta(&self) -> Float {
            self.0.theta()
        }
        #[getter]
        fn phi(&self) -> Float {
            self.0.phi()
        }
        #[getter]
        fn unit(&self) -> Self {
            Self(self.0.unit())
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
    struct Event(rust::data::Event);

    #[pymethods]
    impl Event {
        #[new]
        pub(crate) fn new(p4s: Vec<Vector4>, eps: Vec<Vector3>, weight: Float) -> Self {
            Self(rust::data::Event {
                p4s: p4s.into_iter().map(|arr| arr.0).collect(),
                eps: eps.into_iter().map(|arr| arr.0).collect(),
                weight,
            })
        }
        pub(crate) fn __str__(&self) -> String {
            self.0.to_string()
        }
        #[getter]
        pub(crate) fn get_p4s(&self) -> Vec<Vector4> {
            self.0.p4s.iter().map(|p4| Vector4(*p4)).collect()
        }
        #[setter]
        pub(crate) fn set_p4s(&mut self, value: Vec<Vector4>) {
            self.0.p4s = value.iter().map(|p4| p4.0).collect();
        }
        #[getter]
        pub(crate) fn get_eps(&self) -> Vec<Vector3> {
            self.0.eps.iter().map(|eps_vec| Vector3(*eps_vec)).collect()
        }
        #[setter]
        pub(crate) fn set_eps(&mut self, value: Vec<Vector3>) {
            self.0.eps = value.iter().map(|eps_vec| eps_vec.0).collect();
        }
        #[getter]
        pub(crate) fn get_weight(&self) -> Float {
            self.0.weight
        }
        #[setter]
        pub(crate) fn set_weight(&mut self, value: Float) {
            self.0.weight = value;
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
    }

    #[pyfunction]
    fn open(path: &str) -> PyResult<Dataset> {
        Ok(Dataset(rust::data::open(path).unwrap()))
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
    struct Expression(rust::amplitudes::Expression);

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
        fn register(&mut self, amplitude: &Amplitude) -> AmplitudeID {
            AmplitudeID(self.0.register(amplitude.0.clone()))
        }
        fn load(&mut self, dataset: &Dataset) -> Evaluator {
            Evaluator(self.0.load(&dataset.0))
        }
    }

    #[pyclass]
    struct Evaluator(rust::amplitudes::Evaluator);

    #[pymethods]
    impl Evaluator {
        #[getter]
        fn parameters(&self) -> Vec<String> {
            self.0.parameters()
        }
        fn activate(&mut self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
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
        fn activate_all(&mut self) {
            self.0.activate_all();
        }
        fn deactivate(&mut self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
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
        fn deactivate_all(&mut self) {
            self.0.deactivate_all();
        }
        fn isolate(&mut self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
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
            expression: &Expression,
        ) -> Bound<'py, PyArray1<Complex<Float>>> {
            PyArray1::from_slice_bound(py, &self.0.evaluate(&expression.0, &parameters))
        }
    }

    #[pyclass]
    struct NLL(rust::amplitudes::NLL);

    #[pymethods]
    impl NLL {
        #[new]
        fn new(manager: &Manager, ds_data: &Dataset, ds_mc: &Dataset) -> Self {
            Self(rust::amplitudes::NLL::new(&manager.0, &ds_data.0, &ds_mc.0))
        }
        #[getter]
        fn parameters(&self) -> Vec<String> {
            self.0.parameters()
        }
        fn activate(&mut self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
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
        fn activate_all(&mut self) {
            self.0.activate_all();
        }
        fn deactivate(&mut self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
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
        fn deactivate_all(&mut self) {
            self.0.deactivate_all();
        }
        fn isolate(&mut self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
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
        fn evaluate(&self, expression: &Expression, parameters: Vec<Float>) -> Float {
            self.0.evaluate(&expression.0, &parameters)
        }
        fn project<'py>(
            &self,
            py: Python<'py>,
            expression: &Expression,
            parameters: Vec<Float>,
        ) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice_bound(py, &self.0.project(&expression.0, &parameters))
        }
        #[pyo3(signature = (expression, p0, bounds = None))]
        fn minimize<'py>(
            &self,
            py: Python<'py>,
            expression: &Expression,
            p0: Vec<Float>,
            bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
        ) -> Bound<'py, PyArray1<Float>> {
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
            let status = self.0.minimize(&expression.0, &p0, bounds);
            println!("{}", status);
            PyArray1::from_slice_bound(py, status.x.as_slice())
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
