use std::{array, collections::HashMap};

use indexmap::IndexSet;
use nalgebra::{SMatrix, SVector};
use num::complex::Complex64;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::{
    amplitudes::{AmplitudeID, ParameterLike},
    LadduError, LadduResult,
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
    #[serde(default)]
    active_indices: Vec<usize>,
    /// The set of all registered free parameter names across registered [`Amplitude`]s
    pub free_parameters: IndexSet<String>,
    /// The set of all registered fixed parameter names across registered [`Amplitude`]s
    pub fixed_parameters: IndexSet<String>,
    /// Values of all constants/fixed parameters across registered [`Amplitude`]s
    pub constants: Vec<f64>,
    /// The [`Cache`] for each [`EventData`](`crate::data::EventData`)
    pub caches: Vec<Cache>,
    scalar_cache_names: HashMap<String, usize>,
    complex_scalar_cache_names: HashMap<String, usize>,
    vector_cache_names: HashMap<String, usize>,
    complex_vector_cache_names: HashMap<String, usize>,
    matrix_cache_names: HashMap<String, usize>,
    complex_matrix_cache_names: HashMap<String, usize>,
    cache_size: usize,
    parameter_entries: HashMap<String, ParameterEntry>,
    pub(crate) parameter_overrides: ParameterTransform,
}

/// A single cache entry corresponding to precomputed data for a particular
/// [`EventData`](crate::data::EventData) in a [`Dataset`](crate::data::Dataset).
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

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ParameterEntry {
    id: ParameterID,
    fixed: Option<f64>,
}

/// Parameter transformation instructions applied during re-registration.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct ParameterTransform {
    /// Mapping from old parameter names to new names.
    pub renames: HashMap<String, String>,
    /// Parameters to force fixed with the provided value.
    pub fixed: HashMap<String, f64>,
    /// Parameters to force free (ignore any fixed value).
    pub freed: IndexSet<String>,
}

impl ParameterTransform {
    /// Merge two transforms, with `other` overriding `self` where keys overlap.
    pub fn merged(&self, other: &Self) -> Self {
        let mut merged = self.clone();
        merged.renames.extend(other.renames.clone());
        merged.fixed.extend(other.fixed.clone());
        merged.freed.extend(other.freed.clone());
        merged
    }
}

impl Resources {
    /// Create a new [`Resources`] instance with a parameter transform applied.
    pub fn with_transform(transform: ParameterTransform) -> Self {
        Self {
            parameter_overrides: transform,
            ..Default::default()
        }
    }

    /// The list of free parameter names.
    pub fn free_parameter_names(&self) -> Vec<String> {
        self.free_parameters.iter().cloned().collect()
    }

    /// The list of fixed parameter names.
    pub fn fixed_parameter_names(&self) -> Vec<String> {
        self.fixed_parameters.iter().cloned().collect()
    }

    /// All parameter names (free first, then fixed).
    pub fn parameter_names(&self) -> Vec<String> {
        self.free_parameter_names()
            .into_iter()
            .chain(self.fixed_parameter_names())
            .collect()
    }

    /// Number of free parameters.
    pub fn n_free_parameters(&self) -> usize {
        self.free_parameters.len()
    }

    /// Number of fixed parameters.
    pub fn n_fixed_parameters(&self) -> usize {
        self.fixed_parameters.len()
    }

    /// Total number of parameters.
    pub fn n_parameters(&self) -> usize {
        self.n_free_parameters() + self.n_fixed_parameters()
    }

    fn rebuild_active_indices(&mut self) {
        self.active_indices.clear();
        self.active_indices.extend(
            self.active
                .iter()
                .enumerate()
                .filter_map(|(idx, &is_active)| if is_active { Some(idx) } else { None }),
        );
    }

    pub(crate) fn refresh_active_indices(&mut self) {
        self.rebuild_active_indices();
    }

    /// Return the indices of active amplitudes.
    pub fn active_indices(&self) -> &[usize] {
        &self.active_indices
    }

    #[inline]
    fn set_activation_state(&mut self, name: &str, active: bool) -> Option<bool> {
        self.amplitudes.get(name).map(|amplitude| {
            let idx = amplitude.1;
            let changed = self.active[idx] != active;
            self.active[idx] = active;
            changed
        })
    }
    /// Activate an [`Amplitude`](crate::amplitudes::Amplitude) by name.
    pub fn activate<T: AsRef<str>>(&mut self, name: T) {
        if self
            .set_activation_state(name.as_ref(), true)
            .unwrap_or(false)
        {
            self.rebuild_active_indices();
        }
    }
    /// Activate several [`Amplitude`](crate::amplitudes::Amplitude)s by name.
    pub fn activate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        let mut changed = false;
        for name in names {
            if self
                .set_activation_state(name.as_ref(), true)
                .unwrap_or(false)
            {
                changed = true;
            }
        }
        if changed {
            self.rebuild_active_indices();
        }
    }
    /// Activate an [`Amplitude`](crate::amplitudes::Amplitude) by name, returning an error if it is missing.
    pub fn activate_strict<T: AsRef<str>>(&mut self, name: T) -> LadduResult<()> {
        let name_ref = name.as_ref();
        match self.set_activation_state(name_ref, true) {
            Some(changed) => {
                if changed {
                    self.rebuild_active_indices();
                }
                Ok(())
            }
            None => Err(LadduError::AmplitudeNotFoundError {
                name: name_ref.to_string(),
            }),
        }
    }
    /// Activate several [`Amplitude`](crate::amplitudes::Amplitude)s by name, returning an error if any are missing.
    pub fn activate_many_strict<T: AsRef<str>>(&mut self, names: &[T]) -> LadduResult<()> {
        let mut changed = false;
        for name in names {
            let name_ref = name.as_ref();
            match self.set_activation_state(name_ref, true) {
                Some(state_changed) => {
                    if state_changed {
                        changed = true;
                    }
                }
                None => {
                    return Err(LadduError::AmplitudeNotFoundError {
                        name: name_ref.to_string(),
                    })
                }
            }
        }
        if changed {
            self.rebuild_active_indices();
        }
        Ok(())
    }
    /// Activate all registered [`Amplitude`](crate::amplitudes::Amplitude)s.
    pub fn activate_all(&mut self) {
        let mut changed = false;
        for active in self.active.iter_mut() {
            if !*active {
                *active = true;
                changed = true;
            }
        }
        if changed {
            self.rebuild_active_indices();
        }
    }
    /// Deactivate an [`Amplitude`](crate::amplitudes::Amplitude) by name.
    pub fn deactivate<T: AsRef<str>>(&mut self, name: T) {
        if self
            .set_activation_state(name.as_ref(), false)
            .unwrap_or(false)
        {
            self.rebuild_active_indices();
        }
    }
    /// Deactivate several [`Amplitude`](crate::amplitudes::Amplitude)s by name.
    pub fn deactivate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        let mut changed = false;
        for name in names {
            if self
                .set_activation_state(name.as_ref(), false)
                .unwrap_or(false)
            {
                changed = true;
            }
        }
        if changed {
            self.rebuild_active_indices();
        }
    }
    /// Deactivate an [`Amplitude`](crate::amplitudes::Amplitude) by name, returning an error if it is missing.
    pub fn deactivate_strict<T: AsRef<str>>(&mut self, name: T) -> LadduResult<()> {
        let name_ref = name.as_ref();
        match self.set_activation_state(name_ref, false) {
            Some(changed) => {
                if changed {
                    self.rebuild_active_indices();
                }
                Ok(())
            }
            None => Err(LadduError::AmplitudeNotFoundError {
                name: name_ref.to_string(),
            }),
        }
    }
    /// Deactivate several [`Amplitude`](crate::amplitudes::Amplitude)s by name, returning an error if any are missing.
    pub fn deactivate_many_strict<T: AsRef<str>>(&mut self, names: &[T]) -> LadduResult<()> {
        let mut changed = false;
        for name in names {
            let name_ref = name.as_ref();
            match self.set_activation_state(name_ref, false) {
                Some(state_changed) => {
                    if state_changed {
                        changed = true;
                    }
                }
                None => {
                    return Err(LadduError::AmplitudeNotFoundError {
                        name: name_ref.to_string(),
                    })
                }
            }
        }
        if changed {
            self.rebuild_active_indices();
        }
        Ok(())
    }
    /// Deactivate all registered [`Amplitude`](crate::amplitudes::Amplitude)s.
    pub fn deactivate_all(&mut self) {
        let mut changed = false;
        for active in self.active.iter_mut() {
            if *active {
                *active = false;
                changed = true;
            }
        }
        if changed {
            self.rebuild_active_indices();
        }
    }
    /// Isolate an [`Amplitude`](crate::amplitudes::Amplitude) by name (deactivate the rest).
    pub fn isolate<T: AsRef<str>>(&mut self, name: T) {
        self.deactivate_all();
        self.activate(name);
    }
    /// Isolate an [`Amplitude`](crate::amplitudes::Amplitude) by name (deactivate the rest), returning an error if it is missing.
    pub fn isolate_strict<T: AsRef<str>>(&mut self, name: T) -> LadduResult<()> {
        self.deactivate_all();
        self.activate_strict(name)
    }
    /// Isolate several [`Amplitude`](crate::amplitudes::Amplitude)s by name (deactivate the rest).
    pub fn isolate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        self.deactivate_all();
        self.activate_many(names);
    }
    /// Isolate several [`Amplitude`](crate::amplitudes::Amplitude)s by name (deactivate the rest), returning an error if any are missing.
    pub fn isolate_many_strict<T: AsRef<str>>(&mut self, names: &[T]) -> LadduResult<()> {
        self.deactivate_all();
        self.activate_many_strict(names)
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
    pub fn register_amplitude(&mut self, name: &str) -> LadduResult<AmplitudeID> {
        if self.amplitudes.contains_key(name) {
            return Err(LadduError::RegistrationError {
                name: name.to_string(),
            });
        }
        let next_id = AmplitudeID(name.to_string(), self.amplitudes.len());
        self.amplitudes.insert(name.to_string(), next_id.clone());
        self.active.push(true);
        self.rebuild_active_indices();
        Ok(next_id)
    }

    /// Fetch the [`AmplitudeID`] for a previously registered amplitude by name.
    pub fn amplitude_id(&self, name: &str) -> Option<AmplitudeID> {
        self.amplitudes.get(name).cloned()
    }

    fn apply_transform(&self, name: &str, fixed: Option<f64>) -> (String, Option<f64>) {
        let final_name = self
            .parameter_overrides
            .renames
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string());
        let fixed_value = if let Some(value) = self.parameter_overrides.fixed.get(name) {
            Some(*value)
        } else if self.parameter_overrides.freed.contains(name) {
            None
        } else {
            fixed
        };
        (final_name, fixed_value)
    }

    /// Register a parameter. This method should be called within
    /// [`Amplitude::register`](crate::amplitudes::Amplitude::register). The resulting
    /// [`ParameterID`] should be stored to retrieve the value from the [`Parameters`] wrapper.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter is unnamed, if the name is reused with incompatible
    /// fixed/free status or fixed value, or if renaming causes a conflict.
    pub fn register_parameter(&mut self, p: &ParameterLike) -> LadduResult<ParameterID> {
        let base_name = p.name();
        if base_name.is_empty() {
            return Err(LadduError::UnregisteredParameter {
                name: "<unnamed>".to_string(),
                reason: "Parameter was not initialized with a name".to_string(),
            });
        }
        let (final_name, fixed_value) = self.apply_transform(base_name, p.fixed);

        if let Some(existing) = self.parameter_entries.get(&final_name) {
            match (existing.fixed, fixed_value) {
                (Some(a), Some(b)) if (a - b).abs() > f64::EPSILON => {
                    return Err(LadduError::ParameterConflict {
                        name: final_name,
                        reason: "conflicting fixed values for the same parameter name".to_string(),
                    })
                }
                (Some(_), None) => {
                    return Err(LadduError::ParameterConflict {
                        name: final_name,
                        reason: "attempted to use a fixed parameter name as free".to_string(),
                    })
                }
                (None, Some(_)) => {
                    return Err(LadduError::ParameterConflict {
                        name: final_name,
                        reason: "attempted to use a free parameter name as fixed".to_string(),
                    })
                }
                _ => return Ok(existing.id),
            }
        }

        let entry = if let Some(value) = fixed_value {
            self.fixed_parameters.insert(final_name.clone());
            self.constants.push(value);
            ParameterEntry {
                id: ParameterID::Constant(self.constants.len() - 1),
                fixed: Some(value),
            }
        } else {
            let (index, _) = self.free_parameters.insert_full(final_name.clone());
            ParameterEntry {
                id: ParameterID::Parameter(index),
                fixed: None,
            }
        };
        self.parameter_entries.insert(final_name, entry.clone());
        Ok(entry.id)
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
    use num::complex::Complex64;

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

        resources.deactivate_strict("amp1").unwrap();
        assert!(!resources.active[amp1.1]);
        assert!(resources.active[amp2.1]);

        resources.activate_strict("amp1").unwrap();
        assert!(resources.active[amp1.1]);

        resources.deactivate_all();
        assert!(!resources.active[amp1.1]);
        assert!(!resources.active[amp2.1]);

        resources.activate_all();
        assert!(resources.active[amp1.1]);
        assert!(resources.active[amp2.1]);

        resources.isolate_strict("amp1").unwrap();
        assert!(resources.active[amp1.1]);
        assert!(!resources.active[amp2.1]);
    }

    #[test]
    fn test_resources_parameter_registration() {
        let mut resources = Resources::default();

        let param1 = resources
            .register_parameter(&ParameterLike::free("param1"))
            .unwrap();
        let const1 = resources
            .register_parameter(&ParameterLike::fixed("const1", 1.0))
            .unwrap();

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

        let value1 = Complex64::new(1.0, 2.0);
        let value2 = Complex64::new(3.0, 4.0);
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

        let value1 = Vector2::new(Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0));
        let value2 = Vector2::new(Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0));
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
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
        );
        let value2 = Matrix2::new(
            Complex64::new(9.0, 10.0),
            Complex64::new(11.0, 12.0),
            Complex64::new(13.0, 14.0),
            Complex64::new(15.0, 16.0),
        );
        cache.store_complex_matrix(complex_matrix_id1, value1);
        cache.store_complex_matrix(complex_matrix_id2, value2);

        assert_eq!(cache.get_complex_matrix(complex_matrix_id1), value1);
        assert_eq!(cache.get_complex_matrix(complex_matrix_id2), value2);
        assert_eq!(cache.get_complex_matrix(complex_matrix_id3), value1);
    }

    #[test]
    fn test_uninit_parameter_registration() {
        let mut resources = Resources::default();
        let result = resources.register_parameter(&ParameterLike::uninit());
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_named_amplitude_registration_error() {
        let mut resources = Resources::default();
        assert!(resources.register_amplitude("test_amp").is_ok());
        assert!(resources.register_amplitude("test_amp").is_err());
    }
}
