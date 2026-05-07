use std::{collections::HashMap, fmt::Display, ops::Index};

use serde::{Deserialize, Serialize};

use super::Parameter;
use crate::{
    resources::{ParameterID, Parameters},
    LadduError, LadduResult,
};

/// An ordered set of [`Parameter`]s.
#[derive(Default, Debug, Clone)]
pub struct ParameterMap {
    parameters: Vec<Parameter>,
    name_to_index: HashMap<String, usize>,
}

#[derive(Serialize, Deserialize)]
struct ParameterMapSerde {
    parameters: Vec<Parameter>,
}

impl Index<usize> for ParameterMap {
    type Output = Parameter;

    fn index(&self, index: usize) -> &Self::Output {
        &self.parameters[index]
    }
}

impl Index<&str> for ParameterMap {
    type Output = Parameter;

    fn index(&self, key: &str) -> &Self::Output {
        self.get(key)
            .unwrap_or_else(|| panic!("parameter '{key}' not found"))
    }
}

impl IntoIterator for ParameterMap {
    type Item = Parameter;
    type IntoIter = std::vec::IntoIter<Parameter>;

    fn into_iter(self) -> Self::IntoIter {
        self.parameters.into_iter()
    }
}

impl<'a> IntoIterator for &'a ParameterMap {
    type Item = &'a Parameter;
    type IntoIter = std::slice::Iter<'a, Parameter>;

    fn into_iter(self) -> Self::IntoIter {
        self.parameters.iter()
    }
}

impl Serialize for ParameterMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        ParameterMapSerde {
            parameters: self.parameters.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ParameterMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let serde = ParameterMapSerde::deserialize(deserializer)?;
        Ok(Self::from_parameters(serde.parameters))
    }
}

impl ParameterMap {
    fn from_parameters(parameters: Vec<Parameter>) -> Self {
        let name_to_index = parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| (parameter.name(), index))
            .collect();
        Self {
            parameters,
            name_to_index,
        }
    }

    /// Register a parameter into the ordered map and return its assembled [`ParameterID`].
    pub fn register_parameter(&mut self, p: &Parameter) -> LadduResult<ParameterID> {
        let name = p.name();
        if name.is_empty() {
            return Err(LadduError::UnregisteredParameter {
                name: "<unnamed>".to_string(),
                reason: "Parameter was not initialized with a name".to_string(),
            });
        }

        if let Some((index, existing)) = self.get_indexed(&name) {
            match (existing.fixed(), p.fixed()) {
                (Some(a), Some(b)) if (a - b).abs() > f64::EPSILON => {
                    return Err(LadduError::ParameterConflict {
                        name,
                        reason: "conflicting fixed values for the same parameter name".to_string(),
                    });
                }
                (Some(_), None) => {
                    return Err(LadduError::ParameterConflict {
                        name,
                        reason: "attempted to use a fixed parameter name as free".to_string(),
                    });
                }
                (None, Some(_)) => {
                    return Err(LadduError::ParameterConflict {
                        name,
                        reason: "attempted to use a free parameter name as fixed".to_string(),
                    });
                }
                (Some(_), Some(_)) | (None, None) => return Ok(self.parameter_id(index)),
            }
        }

        let index = self.parameters.len();
        self.insert(p.clone());
        Ok(self.parameter_id(index))
    }

    /// Return the assembled indices of all free parameters.
    pub fn free_parameter_indices(&self) -> Vec<usize> {
        (0..self.free().len()).collect()
    }

    /// Rename a single parameter in place.
    pub fn rename_parameter(&mut self, old: &str, new: &str) -> LadduResult<()> {
        if old == new {
            return Ok(());
        }
        if self.contains_key(new) {
            return Err(LadduError::ParameterConflict {
                name: new.to_string(),
                reason: "rename target already exists".to_string(),
            });
        }
        if let Some(index) = self.index(old) {
            let parameter = self.parameters[index].clone();
            parameter.set_name(new);
            self.name_to_index.remove(old);
            self.name_to_index.insert(new.to_string(), index);
        } else {
            self.assert_parameter_exists(old)?;
        }
        Ok(())
    }

    /// Rename multiple parameters in place.
    pub fn rename_parameters(&mut self, mapping: &HashMap<String, String>) -> LadduResult<()> {
        for (old, new) in mapping {
            self.rename_parameter(old, new)?;
        }
        Ok(())
    }

    /// Fix a parameter to the supplied value.
    pub fn fix_parameter(&self, name: &str, value: f64) -> LadduResult<()> {
        self.assert_parameter_exists(name)?;
        if let Some(parameter) = self.get(name) {
            parameter.set_fixed_value(Some(value));
        }
        Ok(())
    }

    /// Mark a parameter as free.
    pub fn free_parameter(&self, name: &str) -> LadduResult<()> {
        self.assert_parameter_exists(name)?;
        if let Some(parameter) = self.get(name) {
            parameter.set_fixed_value(None);
        }
        Ok(())
    }

    /// Return whether a parameter with the given name exists.
    pub fn contains_key(&self, name: &str) -> bool {
        self.name_to_index.contains_key(name)
    }

    /// Return the storage index for a named parameter.
    pub fn index(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Insert or replace a parameter by name while preserving insertion order.
    pub fn insert(&mut self, parameter: Parameter) -> Option<Parameter> {
        let name = parameter.name();
        if let Some(index) = self.index(&name) {
            Some(std::mem::replace(&mut self.parameters[index], parameter))
        } else {
            let index = self.parameters.len();
            self.parameters.push(parameter);
            self.name_to_index.insert(name, index);
            None
        }
    }

    /// The number of parameters in the set.
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Returns true if the parameter set has no elements.
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Iterate over all parameters in the set.
    pub fn iter(&self) -> std::slice::Iter<'_, Parameter> {
        self.parameters.iter()
    }

    /// Get a parameter by name.
    pub fn get(&self, key: &str) -> Option<&Parameter> {
        self.index(key).map(|index| &self.parameters[index])
    }

    /// Get both the storage index and parameter for a given name.
    pub fn get_indexed(&self, key: &str) -> Option<(usize, &Parameter)> {
        self.index(key)
            .map(|index| (index, &self.parameters[index]))
    }

    /// Get all parameter names in order.
    pub fn names(&self) -> Vec<String> {
        self.parameters.iter().map(Parameter::name).collect()
    }

    /// Filter the parameter set by a predicate.
    pub fn filter(&self, predicate: impl Fn(&Parameter) -> bool) -> Self {
        Self::from_parameters(
            self.parameters
                .iter()
                .filter(|parameter| predicate(parameter))
                .cloned()
                .collect(),
        )
    }

    /// Get a set containing only free parameters.
    pub fn free(&self) -> Self {
        self.filter(|p| p.is_free())
    }

    /// Get a set containing only fixed parameters.
    pub fn fixed(&self) -> Self {
        self.filter(|p| p.is_fixed())
    }

    /// Get a set containing only initialized parameters.
    pub fn initialized(&self) -> Self {
        self.filter(|p| p.initial().is_some())
    }

    /// Get a set containing only uninitialized parameters.
    pub fn uninitialized(&self) -> Self {
        self.filter(|p| p.initial().is_none())
    }

    /// Assemble free inputs into a full [`Parameters`] object.
    ///
    /// The resulting values are ordered with all free parameters first, followed by fixed ones.
    pub fn assemble(&self, free_values: &[f64]) -> LadduResult<Parameters> {
        let expected_free = self.free().len();
        let n_fixed = self.fixed().len();
        let mut values = vec![0.0; expected_free + n_fixed];
        let mut storage_to_assembled = vec![0; self.len()];
        let mut free_iter = free_values.iter();
        let mut free_index = 0;
        let mut fixed_index = expected_free;
        for (storage_index, parameter) in self.parameters.iter().enumerate() {
            if let Some(value) = parameter.fixed() {
                values[fixed_index] = value;
                storage_to_assembled[storage_index] = fixed_index;
                fixed_index += 1;
            } else if let Some(value) = free_iter.next() {
                values[free_index] = *value;
                storage_to_assembled[storage_index] = free_index;
                free_index += 1;
            } else {
                return Err(LadduError::LengthMismatch {
                    context: "parameter values".to_string(),
                    expected: expected_free,
                    actual: free_values.len(),
                });
            }
        }
        if free_iter.next().is_some() {
            return Err(LadduError::LengthMismatch {
                context: "parameter values".to_string(),
                expected: expected_free,
                actual: free_values.len(),
            });
        }
        Ok(Parameters::new(values, expected_free, storage_to_assembled))
    }

    /// Merge two parameter maps.
    ///
    /// When parameters overlap, the state and value stored in `self` always take precedence over
    /// entries from `other`.
    pub fn merge(&self, other: &Self) -> (Self, Vec<usize>, Vec<usize>) {
        let mut merged = self.clone();
        let mut right_map = Vec::with_capacity(other.len());
        for parameter in other {
            let idx = merged.ensure_parameter(parameter.clone());
            right_map.push(idx);
        }
        let left_map: Vec<usize> = (0..self.len())
            .map(|index| merged.assembled_index(index))
            .collect();
        let right_map = right_map
            .into_iter()
            .map(|index| merged.assembled_index(index))
            .collect();
        (merged, left_map, right_map)
    }

    /// Extend a parameter map from another one.
    ///
    /// When both managers reference the same parameter, the value and fixed/free status from
    /// `self` are retained.
    pub fn extend_from(&self, other: &Self) -> (Self, Vec<usize>) {
        let mut merged = self.clone();
        let mut indices = Vec::with_capacity(other.len());
        for parameter in other {
            let idx = merged.ensure_parameter(parameter.clone());
            indices.push(idx);
        }
        let indices = indices
            .into_iter()
            .map(|index| merged.assembled_index(index))
            .collect();
        (merged, indices)
    }

    fn ensure_parameter(&mut self, parameter: Parameter) -> usize {
        let name = parameter.name();
        if let Some(idx) = self.index(&name) {
            return idx;
        }
        let idx = self.len();
        self.insert(parameter);
        idx
    }

    fn assembled_index(&self, storage_index: usize) -> usize {
        let n_free = self
            .parameters
            .iter()
            .filter(|parameter| parameter.is_free())
            .count();
        let preceding_in_group = self.parameters[..storage_index]
            .iter()
            .filter(|parameter| self.parameters[storage_index].is_free() == parameter.is_free())
            .count();
        if self.parameters[storage_index].is_free() {
            preceding_in_group
        } else {
            n_free + preceding_in_group
        }
    }

    fn parameter_id(&self, storage_index: usize) -> ParameterID {
        if self.parameters[storage_index].is_fixed() {
            ParameterID::Constant(storage_index)
        } else {
            ParameterID::Parameter(storage_index)
        }
    }

    fn assert_parameter_exists(&self, name: &str) -> LadduResult<()> {
        if self.contains_key(name) {
            Ok(())
        } else {
            Err(LadduError::UnregisteredParameter {
                name: name.to_string(),
                reason: "parameter not found".to_string(),
            })
        }
    }
}

impl Display for ParameterMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ParameterMap:")?;
        if self.parameters.is_empty() {
            writeln!(f, "  <empty>")?;
            return Ok(());
        }
        writeln!(f, "  free:")?;
        let mut wrote_free = false;
        for parameter in self
            .parameters
            .iter()
            .filter(|parameter| parameter.is_free())
        {
            wrote_free = true;
            writeln!(f, "    {}", parameter.name())?;
        }
        if !wrote_free {
            writeln!(f, "    <none>")?;
        }
        writeln!(f, "  fixed:")?;
        let mut wrote_fixed = false;
        for parameter in self
            .parameters
            .iter()
            .filter(|parameter| parameter.is_fixed())
        {
            wrote_fixed = true;
            if let Some(value) = parameter.fixed() {
                writeln!(f, "    {} = {}", parameter.name(), value)?;
            }
        }
        if !wrote_fixed {
            writeln!(f, "    <none>")?;
        }
        Ok(())
    }
}
