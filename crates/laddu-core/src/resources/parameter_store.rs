use serde::{Deserialize, Serialize};

/// This struct holds references to the constants and free parameters used in the fit so that they
/// may be obtained from their corresponding [`ParameterID`].
#[derive(Debug)]
pub struct Parameters {
    values: Vec<f64>,
    n_free: usize,
    storage_to_assembled: Vec<usize>,
}

impl Parameters {
    /// Create a full parameter store from assembled values and the number of free parameters.
    pub fn new(values: Vec<f64>, n_free: usize, storage_to_assembled: Vec<usize>) -> Self {
        Self {
            values,
            n_free,
            storage_to_assembled,
        }
    }

    /// Borrow the assembled parameter values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Create a new parameter store with the same layout and different assembled values.
    pub fn with_values(&self, values: Vec<f64>) -> Self {
        Self {
            values,
            n_free: self.n_free,
            storage_to_assembled: self.storage_to_assembled.clone(),
        }
    }

    /// Obtain a parameter value or constant value from the given [`ParameterID`].
    pub fn get(&self, pid: ParameterID) -> f64 {
        self.assembled_index(pid)
            .and_then(|index| self.values.get(index))
            .copied()
            .unwrap_or(f64::NAN)
    }

    /// Return the assembled index for the given registered parameter.
    pub fn assembled_index(&self, pid: ParameterID) -> Option<usize> {
        self.storage_index(pid)
            .and_then(|index| self.storage_to_assembled.get(index).copied())
    }

    /// Return the free-parameter index for the given registered parameter, if it is currently
    /// free.
    pub fn free_index(&self, pid: ParameterID) -> Option<usize> {
        let index = self.assembled_index(pid)?;
        (index < self.n_free).then_some(index)
    }

    fn storage_index(&self, pid: ParameterID) -> Option<usize> {
        match pid {
            ParameterID::Parameter(index) | ParameterID::Constant(index) => Some(index),
            ParameterID::Uninit => None,
        }
    }

    /// Return the number of free parameters.
    pub fn len(&self) -> usize {
        self.n_free
    }

    /// Return whether there are no free parameters.
    pub fn is_empty(&self) -> bool {
        self.n_free == 0
    }
}

/// An object which acts as a tag to refer to either a free parameter or a constant value.
#[derive(Default, Copy, Clone, Debug, Serialize, Deserialize)]
pub enum ParameterID {
    /// A free parameter.
    Parameter(usize),
    /// A constant value.
    Constant(usize),
    /// An uninitialized ID.
    #[default]
    Uninit,
}
