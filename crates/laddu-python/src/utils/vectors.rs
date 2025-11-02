use laddu_core::{Vec3, Vec4};
use pyo3::{exceptions::PyTypeError, prelude::*};
use pyo3_polars::PyExpr;

#[pyclass(name = "Vec3", module = "laddu")]
#[derive(Clone)]
pub struct PyVec3(Vec3);

#[pymethods]
impl PyVec3 {
    #[new]
    fn new(name: String) -> Self {
        Self(Vec3::new(name))
    }
    fn alias(&self, name: String) -> [PyExpr; 3] {
        let [x, y, z] = self.0.alias(name);
        [PyExpr(x), PyExpr(y), PyExpr(z)]
    }
    fn x(&self) -> PyExpr {
        PyExpr(self.0.x())
    }
    fn y(&self) -> PyExpr {
        PyExpr(self.0.y())
    }
    fn z(&self) -> PyExpr {
        PyExpr(self.0.z())
    }
    fn with_mass(&self, mass: Bound<PyAny>) -> PyResult<PyVec4> {
        Ok(PyVec4(self.0.with_mass(&mass.extract::<PyExpr>()?.0)))
    }
    fn with_energy(&self, energy: Bound<PyAny>) -> PyResult<PyVec4> {
        Ok(PyVec4(self.0.with_energy(&energy.extract::<PyExpr>()?.0)))
    }
    fn dot(&self, other: &Self) -> PyExpr {
        PyExpr(self.0.dot(&other.0))
    }
    fn cross(&self, other: &Self) -> Self {
        Self(self.0.cross(&other.0))
    }
    fn mag2(&self) -> PyExpr {
        PyExpr(self.0.mag2())
    }
    fn mag(&self) -> PyExpr {
        PyExpr(self.0.mag())
    }
    fn costheta(&self) -> PyExpr {
        PyExpr(self.0.costheta())
    }
    fn theta(&self) -> PyExpr {
        PyExpr(self.0.theta())
    }
    fn phi(&self) -> PyExpr {
        PyExpr(self.0.phi())
    }
    fn unit(&self) -> Self {
        Self(self.0.unit())
    }
    fn add(&self, other: &Self) -> Self {
        Self(self.0.add(&other.0))
    }
    fn scalar_add(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        // TODO: better error message
        Ok(Self(self.0.scalar_add(&other.extract::<PyExpr>()?.0)))
    }
    fn sub(&self, other: &Self) -> Self {
        Self(self.0.sub(&other.0))
    }
    fn scalar_sub(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        // TODO: better error message
        Ok(Self(self.0.scalar_sub(&other.extract::<PyExpr>()?.0)))
    }
    fn scalar_rsub(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        // TODO: better error message
        Ok(Self(self.0.scalar_rsub(&other.extract::<PyExpr>()?.0)))
    }
    fn mul(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        // TODO: better error message
        Ok(Self(self.0.mul(&other.extract::<PyExpr>()?.0)))
    }
    fn div(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        // TODO: better error message
        Ok(Self(self.0.div(&other.extract::<PyExpr>()?.0)))
    }
    fn rdiv(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        // TODO: better error message
        Ok(Self(self.0.rdiv(&other.extract::<PyExpr>()?.0)))
    }
    fn neg(&self) -> Self {
        Self(self.0.neg())
    }

    fn __add__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        if other.extract::<PyExpr>().is_ok() {
            self.scalar_add(other)
        } else if let Ok(vector) = other.extract::<Self>() {
            Ok(self.add(&vector))
        } else {
            Err(PyTypeError::new_err(format!(
                "Unsupported operand type(s) for +: 'Vec3' and {}",
                other.get_type().name()?
            )))
        }
    }
    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }
    fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        if other.extract::<PyExpr>().is_ok() {
            self.scalar_sub(other)
        } else if let Ok(vector) = other.extract::<Self>() {
            Ok(self.sub(&vector))
        } else {
            Err(PyTypeError::new_err(format!(
                "Unsupported operand type(s) for -: 'Vec3' and {}",
                other.get_type().name()?
            )))
        }
    }
    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        if other.extract::<PyExpr>().is_ok() {
            self.scalar_rsub(other)
        } else if let Ok(vector) = other.extract::<Self>() {
            Ok(vector.sub(self))
        } else {
            Err(PyTypeError::new_err(format!(
                "Unsupported operand type(s) for -: 'Vec3' and {}",
                other.get_type().name()?
            )))
        }
    }
    fn __mul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.mul(other)
    }
    fn __rmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.mul(other)
    }
    fn __truediv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.div(other)
    }
    fn __rtruediv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.rdiv(other)
    }
    fn __neg__(&self) -> Self {
        self.neg()
    }
}

#[pyclass(name = "Vec4", module = "laddu")]
#[derive(Clone)]
pub struct PyVec4(Vec4);

#[pymethods]
impl PyVec4 {
    #[new]
    fn new(name: String) -> Self {
        Self(Vec4::new(name))
    }
    #[staticmethod]
    fn sum(constituents: Vec<String>) -> Self {
        Self(Vec4::sum(constituents))
    }
    fn alias(&self, name: String) -> Vec<PyExpr> {
        self.0
            .alias(name)
            .iter()
            .map(|e| PyExpr(e.clone()))
            .collect()
    }
    fn px(&self) -> PyExpr {
        PyExpr(self.0.px())
    }
    fn py(&self) -> PyExpr {
        PyExpr(self.0.py())
    }
    fn pz(&self) -> PyExpr {
        PyExpr(self.0.pz())
    }
    fn e(&self) -> PyExpr {
        PyExpr(self.0.e())
    }
    fn vec3(&self) -> PyVec3 {
        PyVec3(self.0.vec3())
    }
    fn beta(&self) -> PyVec3 {
        PyVec3(self.0.beta())
    }
    fn gamma(&self) -> PyExpr {
        PyExpr(self.0.gamma())
    }
    fn mag2(&self) -> PyExpr {
        PyExpr(self.0.mag2())
    }
    fn mag(&self) -> PyExpr {
        PyExpr(self.0.mag())
    }
    fn boost(&self, beta: PyVec3) -> Self {
        Self(self.0.boost(&beta.0))
    }
    fn add(&self, other: &Self) -> Self {
        Self(self.0.add(&other.0))
    }
    fn sub(&self, other: &Self) -> Self {
        Self(self.0.sub(&other.0))
    }
    fn neg(&self) -> Self {
        Self(self.0.neg())
    }
    fn __add__(&self, other: &Self) -> Self {
        Self(self.0.add(&other.0))
    }
    fn __sub__(&self, other: &Self) -> Self {
        Self(self.0.sub(&other.0))
    }
    fn __neg__(&self) -> Self {
        self.neg()
    }
}
