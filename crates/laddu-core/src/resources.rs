use std::{array, collections::HashMap};

use indexmap::IndexSet;
use nalgebra::{SMatrix, SVector};
use num::complex::Complex64;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::{
    amplitudes::{AmplitudeID, ParameterLike},
    LadduError,
};

/// This struct holds references to the constants and free parameters used in the fit so that they
/// may be obtained from their corresponding [`ParameterID`].
#[derive(Debug)]
pub struct Parameters<'a> {
    pub(crate) parameters: &'a [f64],
    pub(crate) constants: &'a [f64],
}

impl<'a> Parameters<'a> {
    /// Create a new set of [`Parameters`] from a list of floating values and a list of constant values
    pub fn new(parameters: &'a [f64], constants: &'a [f64]) -> Self {
        Self {
            parameters,
            constants,
        }
    }

    /// Obtain a parameter value or constant value from the given [`ParameterID`].
    pub fn get(&self, pid: ParameterID) -> f64 {
        match pid {
            ParameterID::Parameter(index) => self.parameters[index],
            ParameterID::Constant(index) => self.constants[index],
            ParameterID::Uninit => panic!("Parameter has not been registered!"),
        }
    }

    /// The number of free parameters.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.parameters.len()
    }
}

/// The main resource manager for cached values, amplitudes, parameters, and constants.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Resources {
    amplitudes: HashMap<String, AmplitudeID>,
    /// A list indicating which amplitudes are active (using [`AmplitudeID`]s as indices)
    pub active: Vec<bool>,
    /// The set of all registered parameter names across registered [`Amplitude`](`crate::amplitudes::Amplitude`)s
    pub parameters: IndexSet<String>,
    /// Values of all constants across registered [`Amplitude`](`crate::amplitudes::Amplitude`)s
    pub constants: Vec<f64>,
    /// The [`Cache`] for each [`Event`](`crate::Event`)
    pub caches: Vec<Cache>,
    scalar_cache_names: HashMap<String, usize>,
    complex_scalar_cache_names: HashMap<String, usize>,
    vector_cache_names: HashMap<String, usize>,
    complex_vector_cache_names: HashMap<String, usize>,
    matrix_cache_names: HashMap<String, usize>,
    complex_matrix_cache_names: HashMap<String, usize>,
    cache_size: usize,
}

/// A single cache entry corresponding to precomputed data for a particular
/// [`Event`](crate::data::Event) in a [`Dataset`](crate::data::Dataset).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cache(Vec<f64>);
impl Cache {
    fn new(cache_size: usize) -> Self {
        Self(vec![0.0; cache_size])
    }
    /// Store a scalar value with the corresponding [`ScalarID`].
    pub fn store_scalar(&mut self, sid: ScalarID, value: f64) {
        self.0[sid.0] = value;
    }
    /// Store a complex scalar value with the corresponding [`ComplexScalarID`].
    pub fn store_complex_scalar(&mut self, csid: ComplexScalarID, value: Complex64) {
        self.0[csid.0] = value.re;
        self.0[csid.1] = value.im;
    }
    /// Store a vector with the corresponding [`VectorID`].
    pub fn store_vector<const R: usize>(&mut self, vid: VectorID<R>, value: SVector<f64, R>) {
        vid.0
            .into_iter()
            .enumerate()
            .for_each(|(vi, i)| self.0[i] = value[vi]);
    }
    /// Store a complex-valued vector with the corresponding [`ComplexVectorID`].
    pub fn store_complex_vector<const R: usize>(
        &mut self,
        cvid: ComplexVectorID<R>,
        value: SVector<Complex64, R>,
    ) {
        cvid.0
            .into_iter()
            .enumerate()
            .for_each(|(vi, i)| self.0[i] = value[vi].re);
        cvid.1
            .into_iter()
            .enumerate()
            .for_each(|(vi, i)| self.0[i] = value[vi].im);
    }
    /// Store a matrix with the corresponding [`MatrixID`].
    pub fn store_matrix<const R: usize, const C: usize>(
        &mut self,
        mid: MatrixID<R, C>,
        value: SMatrix<f64, R, C>,
    ) {
        mid.0.into_iter().enumerate().for_each(|(vi, row)| {
            row.into_iter()
                .enumerate()
                .for_each(|(vj, k)| self.0[k] = value[(vi, vj)])
        });
    }
    /// Store a complex-valued matrix with the corresponding [`ComplexMatrixID`].
    pub fn store_complex_matrix<const R: usize, const C: usize>(
        &mut self,
        cmid: ComplexMatrixID<R, C>,
        value: SMatrix<Complex64, R, C>,
    ) {
        cmid.0.into_iter().enumerate().for_each(|(vi, row)| {
            row.into_iter()
                .enumerate()
                .for_each(|(vj, k)| self.0[k] = value[(vi, vj)].re)
        });
        cmid.1.into_iter().enumerate().for_each(|(vi, row)| {
            row.into_iter()
                .enumerate()
                .for_each(|(vj, k)| self.0[k] = value[(vi, vj)].im)
        });
    }
    /// Retrieve a scalar value from the [`Cache`].
    pub fn get_scalar(&self, sid: ScalarID) -> f64 {
        self.0[sid.0]
    }
    /// Retrieve a complex scalar value from the [`Cache`].
    pub fn get_complex_scalar(&self, csid: ComplexScalarID) -> Complex64 {
        Complex64::new(self.0[csid.0], self.0[csid.1])
    }
    /// Retrieve a vector from the [`Cache`].
    pub fn get_vector<const R: usize>(&self, vid: VectorID<R>) -> SVector<f64, R> {
        SVector::from_fn(|i, _| self.0[vid.0[i]])
    }
    /// Retrieve a complex-valued vector from the [`Cache`].
    pub fn get_complex_vector<const R: usize>(
        &self,
        cvid: ComplexVectorID<R>,
    ) -> SVector<Complex64, R> {
        SVector::from_fn(|i, _| Complex64::new(self.0[cvid.0[i]], self.0[cvid.1[i]]))
    }
    /// Retrieve a matrix from the [`Cache`].
    pub fn get_matrix<const R: usize, const C: usize>(
        &self,
        mid: MatrixID<R, C>,
    ) -> SMatrix<f64, R, C> {
        SMatrix::from_fn(|i, j| self.0[mid.0[i][j]])
    }
    /// Retrieve a complex-valued matrix from the [`Cache`].
    pub fn get_complex_matrix<const R: usize, const C: usize>(
        &self,
        cmid: ComplexMatrixID<R, C>,
    ) -> SMatrix<Complex64, R, C> {
        SMatrix::from_fn(|i, j| Complex64::new(self.0[cmid.0[i][j]], self.0[cmid.1[i][j]]))
    }
}

/// An object which acts as a tag to refer to either a free parameter or a constant value.
#[derive(Default, Copy, Clone, Debug, Serialize, Deserialize)]
pub enum ParameterID {
    /// A free parameter.
    Parameter(usize),
    /// A constant value.
    Constant(usize),
    /// An uninitialized ID
    #[default]
    Uninit,
}

/// A tag for retrieving or storing a scalar value in a [`Cache`].
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct ScalarID(usize);

/// A tag for retrieving or storing a complex scalar value in a [`Cache`].
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct ComplexScalarID(usize, usize);

/// A tag for retrieving or storing a vector in a [`Cache`].
#[serde_as]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct VectorID<const R: usize>(#[serde_as(as = "[_; R]")] [usize; R]);

impl<const R: usize> Default for VectorID<R> {
    fn default() -> Self {
        Self([0; R])
    }
}

/// A tag for retrieving or storing a complex-valued vector in a [`Cache`].
#[serde_as]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct ComplexVectorID<const R: usize>(
    #[serde_as(as = "[_; R]")] [usize; R],
    #[serde_as(as = "[_; R]")] [usize; R],
);

impl<const R: usize> Default for ComplexVectorID<R> {
    fn default() -> Self {
        Self([0; R], [0; R])
    }
}

/// A tag for retrieving or storing a matrix in a [`Cache`].
#[serde_as]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct MatrixID<const R: usize, const C: usize>(
    #[serde_as(as = "[[_; C]; R]")] [[usize; C]; R],
);

impl<const R: usize, const C: usize> Default for MatrixID<R, C> {
    fn default() -> Self {
        Self([[0; C]; R])
    }
}

/// A tag for retrieving or storing a complex-valued matrix in a [`Cache`].
#[serde_as]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct ComplexMatrixID<const R: usize, const C: usize>(
    #[serde_as(as = "[[_; C]; R]")] [[usize; C]; R],
    #[serde_as(as = "[[_; C]; R]")] [[usize; C]; R],
);

impl<const R: usize, const C: usize> Default for ComplexMatrixID<R, C> {
    fn default() -> Self {
        Self([[0; C]; R], [[0; C]; R])
    }
}

impl Resources {
    /// Activate an [`Amplitude`](crate::amplitudes::Amplitude) by name.
    pub fn activate<T: AsRef<str>>(&mut self, name: T) -> Result<(), LadduError> {
        self.active[self
            .amplitudes
            .get(name.as_ref())
            .ok_or(LadduError::AmplitudeNotFoundError {
                name: name.as_ref().to_string(),
            })?
            .1] = true;
        Ok(())
    }
    /// Activate several [`Amplitude`](crate::amplitudes::Amplitude)s by name.
    pub fn activate_many<T: AsRef<str>>(&mut self, names: &[T]) -> Result<(), LadduError> {
        for name in names {
            self.activate(name)?
        }
        Ok(())
    }
    /// Activate all registered [`Amplitude`](crate::amplitudes::Amplitude)s.
    pub fn activate_all(&mut self) {
        self.active = vec![true; self.active.len()];
    }
    /// Deactivate an [`Amplitude`](crate::amplitudes::Amplitude) by name.
    pub fn deactivate<T: AsRef<str>>(&mut self, name: T) -> Result<(), LadduError> {
        self.active[self
            .amplitudes
            .get(name.as_ref())
            .ok_or(LadduError::AmplitudeNotFoundError {
                name: name.as_ref().to_string(),
            })?
            .1] = false;
        Ok(())
    }
    /// Deactivate several [`Amplitude`](crate::amplitudes::Amplitude)s by name.
    pub fn deactivate_many<T: AsRef<str>>(&mut self, names: &[T]) -> Result<(), LadduError> {
        for name in names {
            self.deactivate(name)?;
        }
        Ok(())
    }
    /// Deactivate all registered [`Amplitude`](crate::amplitudes::Amplitude)s.
    pub fn deactivate_all(&mut self) {
        self.active = vec![false; self.active.len()];
    }
    /// Isolate an [`Amplitude`](crate::amplitudes::Amplitude) by name (deactivate the rest).
    pub fn isolate<T: AsRef<str>>(&mut self, name: T) -> Result<(), LadduError> {
        self.deactivate_all();
        self.activate(name)
    }
    /// Isolate several [`Amplitude`](crate::amplitudes::Amplitude)s by name (deactivate the rest).
    pub fn isolate_many<T: AsRef<str>>(&mut self, names: &[T]) -> Result<(), LadduError> {
        self.deactivate_all();
        self.activate_many(names)
    }
    /// Register an [`Amplitude`](crate::amplitudes::Amplitude) with the [`Resources`] manager.
    /// This method should be called at the end of the
    /// [`Amplitude::register`](crate::amplitudes::Amplitude::register) method. The
    /// [`Amplitude`](crate::amplitudes::Amplitude) should probably obtain a name [`String`] in its
    /// constructor.
    ///
    /// # Errors
    ///
    /// The [`Amplitude`](crate::amplitudes::Amplitude)'s name must be unique and not already
    /// registered, else this will return a [`RegistrationError`][LadduError::RegistrationError].
    pub fn register_amplitude(&mut self, name: &str) -> Result<AmplitudeID, LadduError> {
        if self.amplitudes.contains_key(name) {
            return Err(LadduError::RegistrationError {
                name: name.to_string(),
            });
        }
        let next_id = AmplitudeID(name.to_string(), self.amplitudes.len());
        self.amplitudes.insert(name.to_string(), next_id.clone());
        self.active.push(true);
        Ok(next_id)
    }
    /// Register a free parameter (or constant) [`ParameterLike`]. This method should be called
    /// within the [`Amplitude::register`](crate::amplitudes::Amplitude::register) method, and the
    /// resulting [`ParameterID`] should be stored to use later to retrieve the value from the
    /// [`Parameters`] wrapper object.
    pub fn register_parameter(&mut self, pl: &ParameterLike) -> ParameterID {
        match pl {
            ParameterLike::Parameter(name) => {
                let (index, _) = self.parameters.insert_full(name.to_string());
                ParameterID::Parameter(index)
            }
            ParameterLike::Constant(value) => {
                self.constants.push(*value);
                ParameterID::Constant(self.constants.len() - 1)
            }
            ParameterLike::Uninit => panic!("Parameter was not initialized!"),
        }
    }
    pub(crate) fn reserve_cache(&mut self, num_events: usize) {
        self.caches = vec![Cache::new(self.cache_size); num_events]
    }
    /// Register a scalar with an optional name (names are unique to the [`Cache`] so two different
    /// registrations of the same type which share a name will also share values and may overwrite
    /// each other). This method should be called within the
    /// [`Amplitude::register`](crate::amplitudes::Amplitude::register) method, and the
    /// resulting [`ScalarID`] should be stored to use later to retrieve the value from the [`Cache`].
    pub fn register_scalar(&mut self, name: Option<&str>) -> ScalarID {
        let first_index = if let Some(name) = name {
            *self
                .scalar_cache_names
                .entry(name.to_string())
                .or_insert_with(|| {
                    self.cache_size += 1;
                    self.cache_size - 1
                })
        } else {
            self.cache_size += 1;
            self.cache_size - 1
        };
        ScalarID(first_index)
    }
    /// Register a complex scalar with an optional name (names are unique to the [`Cache`] so two different
    /// registrations of the same type which share a name will also share values and may overwrite
    /// each other). This method should be called within the
    /// [`Amplitude::register`](crate::amplitudes::Amplitude::register) method, and the
    /// resulting [`ComplexScalarID`] should be stored to use later to retrieve the value from the [`Cache`].
    pub fn register_complex_scalar(&mut self, name: Option<&str>) -> ComplexScalarID {
        let first_index = if let Some(name) = name {
            *self
                .complex_scalar_cache_names
                .entry(name.to_string())
                .or_insert_with(|| {
                    self.cache_size += 2;
                    self.cache_size - 2
                })
        } else {
            self.cache_size += 2;
            self.cache_size - 2
        };
        ComplexScalarID(first_index, first_index + 1)
    }
    /// Register a vector with an optional name (names are unique to the [`Cache`] so two different
    /// registrations of the same type which share a name will also share values and may overwrite
    /// each other). This method should be called within the
    /// [`Amplitude::register`](crate::amplitudes::Amplitude::register) method, and the
    /// resulting [`VectorID`] should be stored to use later to retrieve the value from the [`Cache`].
    pub fn register_vector<const R: usize>(&mut self, name: Option<&str>) -> VectorID<R> {
        let first_index = if let Some(name) = name {
            *self
                .vector_cache_names
                .entry(name.to_string())
                .or_insert_with(|| {
                    self.cache_size += R;
                    self.cache_size - R
                })
        } else {
            self.cache_size += R;
            self.cache_size - R
        };
        VectorID(array::from_fn(|i| first_index + i))
    }
    /// Register a complex-valued vector with an optional name (names are unique to the [`Cache`] so two different
    /// registrations of the same type which share a name will also share values and may overwrite
    /// each other). This method should be called within the
    /// [`Amplitude::register`](crate::amplitudes::Amplitude::register) method, and the
    /// resulting [`ComplexVectorID`] should be stored to use later to retrieve the value from the [`Cache`].
    pub fn register_complex_vector<const R: usize>(
        &mut self,
        name: Option<&str>,
    ) -> ComplexVectorID<R> {
        let first_index = if let Some(name) = name {
            *self
                .complex_vector_cache_names
                .entry(name.to_string())
                .or_insert_with(|| {
                    self.cache_size += R * 2;
                    self.cache_size - (R * 2)
                })
        } else {
            self.cache_size += R * 2;
            self.cache_size - (R * 2)
        };
        ComplexVectorID(
            array::from_fn(|i| first_index + i),
            array::from_fn(|i| (first_index + R) + i),
        )
    }
    /// Register a matrix with an optional name (names are unique to the [`Cache`] so two different
    /// registrations of the same type which share a name will also share values and may overwrite
    /// each other). This method should be called within the
    /// [`Amplitude::register`](crate::amplitudes::Amplitude::register) method, and the
    /// resulting [`MatrixID`] should be stored to use later to retrieve the value from the [`Cache`].
    pub fn register_matrix<const R: usize, const C: usize>(
        &mut self,
        name: Option<&str>,
    ) -> MatrixID<R, C> {
        let first_index = if let Some(name) = name {
            *self
                .matrix_cache_names
                .entry(name.to_string())
                .or_insert_with(|| {
                    self.cache_size += R * C;
                    self.cache_size - (R * C)
                })
        } else {
            self.cache_size += R * C;
            self.cache_size - (R * C)
        };
        MatrixID(array::from_fn(|i| {
            array::from_fn(|j| first_index + i * C + j)
        }))
    }
    /// Register a complex-valued matrix with an optional name (names are unique to the [`Cache`] so two different
    /// registrations of the same type which share a name will also share values and may overwrite
    /// each other). This method should be called within the
    /// [`Amplitude::register`](crate::amplitudes::Amplitude::register) method, and the
    /// resulting [`ComplexMatrixID`] should be stored to use later to retrieve the value from the [`Cache`].
    pub fn register_complex_matrix<const R: usize, const C: usize>(
        &mut self,
        name: Option<&str>,
    ) -> ComplexMatrixID<R, C> {
        let first_index = if let Some(name) = name {
            *self
                .complex_matrix_cache_names
                .entry(name.to_string())
                .or_insert_with(|| {
                    self.cache_size += 2 * R * C;
                    self.cache_size - (2 * R * C)
                })
        } else {
            self.cache_size += 2 * R * C;
            self.cache_size - (2 * R * C)
        };
        ComplexMatrixID(
            array::from_fn(|i| array::from_fn(|j| first_index + i * C + j)),
            array::from_fn(|i| array::from_fn(|j| (first_index + R * C) + i * C + j)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2, Vector2};
    use num::Complex;

    #[test]
    fn test_parameters() {
        let parameters = vec![1.0, 2.0, 3.0];
        let constants = vec![4.0, 5.0, 6.0];
        let params = Parameters::new(&parameters, &constants);

        assert_eq!(params.get(ParameterID::Parameter(0)), 1.0);
        assert_eq!(params.get(ParameterID::Parameter(1)), 2.0);
        assert_eq!(params.get(ParameterID::Parameter(2)), 3.0);
        assert_eq!(params.get(ParameterID::Constant(0)), 4.0);
        assert_eq!(params.get(ParameterID::Constant(1)), 5.0);
        assert_eq!(params.get(ParameterID::Constant(2)), 6.0);
        assert_eq!(params.len(), 3);
    }

    #[test]
    #[should_panic(expected = "Parameter has not been registered!")]
    fn test_uninit_parameter() {
        let parameters = vec![1.0];
        let constants = vec![1.0];
        let params = Parameters::new(&parameters, &constants);
        params.get(ParameterID::Uninit);
    }

    #[test]
    fn test_resources_amplitude_management() {
        let mut resources = Resources::default();

        let amp1 = resources.register_amplitude("amp1").unwrap();
        let amp2 = resources.register_amplitude("amp2").unwrap();

        assert!(resources.active[amp1.1]);
        assert!(resources.active[amp2.1]);

        resources.deactivate("amp1").unwrap();
        assert!(!resources.active[amp1.1]);
        assert!(resources.active[amp2.1]);

        resources.activate("amp1").unwrap();
        assert!(resources.active[amp1.1]);

        resources.deactivate_all();
        assert!(!resources.active[amp1.1]);
        assert!(!resources.active[amp2.1]);

        resources.activate_all();
        assert!(resources.active[amp1.1]);
        assert!(resources.active[amp2.1]);

        resources.isolate("amp1").unwrap();
        assert!(resources.active[amp1.1]);
        assert!(!resources.active[amp2.1]);
    }

    #[test]
    fn test_resources_parameter_registration() {
        let mut resources = Resources::default();

        let param1 = resources.register_parameter(&ParameterLike::Parameter("param1".to_string()));
        let const1 = resources.register_parameter(&ParameterLike::Constant(1.0));

        match param1 {
            ParameterID::Parameter(idx) => assert_eq!(idx, 0),
            _ => panic!("Expected Parameter variant"),
        }

        match const1 {
            ParameterID::Constant(idx) => assert_eq!(idx, 0),
            _ => panic!("Expected Constant variant"),
        }
    }

    #[test]
    fn test_cache_scalar_operations() {
        let mut resources = Resources::default();

        let scalar1 = resources.register_scalar(Some("test_scalar"));
        let scalar2 = resources.register_scalar(None);
        let scalar3 = resources.register_scalar(Some("test_scalar"));

        resources.reserve_cache(1);
        let cache = &mut resources.caches[0];

        cache.store_scalar(scalar1, 1.0);
        cache.store_scalar(scalar2, 2.0);

        assert_eq!(cache.get_scalar(scalar1), 1.0);
        assert_eq!(cache.get_scalar(scalar2), 2.0);
        assert_eq!(cache.get_scalar(scalar3), 1.0);
    }

    #[test]
    fn test_cache_complex_operations() {
        let mut resources = Resources::default();

        let complex1 = resources.register_complex_scalar(Some("test_complex"));
        let complex2 = resources.register_complex_scalar(None);
        let complex3 = resources.register_complex_scalar(Some("test_complex"));

        resources.reserve_cache(1);
        let cache = &mut resources.caches[0];

        let value1 = Complex::new(1.0, 2.0);
        let value2 = Complex::new(3.0, 4.0);
        cache.store_complex_scalar(complex1, value1);
        cache.store_complex_scalar(complex2, value2);

        assert_eq!(cache.get_complex_scalar(complex1), value1);
        assert_eq!(cache.get_complex_scalar(complex2), value2);
        assert_eq!(cache.get_complex_scalar(complex3), value1);
    }

    #[test]
    fn test_cache_vector_operations() {
        let mut resources = Resources::default();

        let vector_id1: VectorID<2> = resources.register_vector(Some("test_vector"));
        let vector_id2: VectorID<2> = resources.register_vector(None);
        let vector_id3: VectorID<2> = resources.register_vector(Some("test_vector"));

        resources.reserve_cache(1);
        let cache = &mut resources.caches[0];

        let value1 = Vector2::new(1.0, 2.0);
        let value2 = Vector2::new(3.0, 4.0);
        cache.store_vector(vector_id1, value1);
        cache.store_vector(vector_id2, value2);

        assert_eq!(cache.get_vector(vector_id1), value1);
        assert_eq!(cache.get_vector(vector_id2), value2);
        assert_eq!(cache.get_vector(vector_id3), value1);
    }

    #[test]
    fn test_cache_complex_vector_operations() {
        let mut resources = Resources::default();

        let complex_vector_id1: ComplexVectorID<2> =
            resources.register_complex_vector(Some("test_complex_vector"));
        let complex_vector_id2: ComplexVectorID<2> = resources.register_complex_vector(None);
        let complex_vector_id3: ComplexVectorID<2> =
            resources.register_complex_vector(Some("test_complex_vector"));

        resources.reserve_cache(1);
        let cache = &mut resources.caches[0];

        let value1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
        let value2 = Vector2::new(Complex::new(5.0, 6.0), Complex::new(7.0, 8.0));
        cache.store_complex_vector(complex_vector_id1, value1);
        cache.store_complex_vector(complex_vector_id2, value2);

        assert_eq!(cache.get_complex_vector(complex_vector_id1), value1);
        assert_eq!(cache.get_complex_vector(complex_vector_id2), value2);
        assert_eq!(cache.get_complex_vector(complex_vector_id3), value1);
    }

    #[test]
    fn test_cache_matrix_operations() {
        let mut resources = Resources::default();

        let matrix_id1: MatrixID<2, 2> = resources.register_matrix(Some("test_matrix"));
        let matrix_id2: MatrixID<2, 2> = resources.register_matrix(None);
        let matrix_id3: MatrixID<2, 2> = resources.register_matrix(Some("test_matrix"));

        resources.reserve_cache(1);
        let cache = &mut resources.caches[0];

        let value1 = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        let value2 = Matrix2::new(5.0, 6.0, 7.0, 8.0);
        cache.store_matrix(matrix_id1, value1);
        cache.store_matrix(matrix_id2, value2);

        assert_eq!(cache.get_matrix(matrix_id1), value1);
        assert_eq!(cache.get_matrix(matrix_id2), value2);
        assert_eq!(cache.get_matrix(matrix_id3), value1);
    }

    #[test]
    fn test_cache_complex_matrix_operations() {
        let mut resources = Resources::default();

        let complex_matrix_id1: ComplexMatrixID<2, 2> =
            resources.register_complex_matrix(Some("test_complex_matrix"));
        let complex_matrix_id2: ComplexMatrixID<2, 2> = resources.register_complex_matrix(None);
        let complex_matrix_id3: ComplexMatrixID<2, 2> =
            resources.register_complex_matrix(Some("test_complex_matrix"));

        resources.reserve_cache(1);
        let cache = &mut resources.caches[0];

        let value1 = Matrix2::new(
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        );
        let value2 = Matrix2::new(
            Complex::new(9.0, 10.0),
            Complex::new(11.0, 12.0),
            Complex::new(13.0, 14.0),
            Complex::new(15.0, 16.0),
        );
        cache.store_complex_matrix(complex_matrix_id1, value1);
        cache.store_complex_matrix(complex_matrix_id2, value2);

        assert_eq!(cache.get_complex_matrix(complex_matrix_id1), value1);
        assert_eq!(cache.get_complex_matrix(complex_matrix_id2), value2);
        assert_eq!(cache.get_complex_matrix(complex_matrix_id3), value1);
    }

    #[test]
    #[should_panic(expected = "Parameter was not initialized!")]
    fn test_uninit_parameter_registration() {
        let mut resources = Resources::default();
        resources.register_parameter(&ParameterLike::Uninit);
    }

    #[test]
    fn test_duplicate_named_amplitude_registration_error() {
        let mut resources = Resources::default();
        assert!(resources.register_amplitude("test_amp").is_ok());
        assert!(resources.register_amplitude("test_amp").is_err());
    }
}
