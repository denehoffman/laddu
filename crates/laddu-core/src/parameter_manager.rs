use std::collections::HashMap;

use indexmap::IndexSet;
use serde::{Deserialize, Serialize};

use crate::{resources::Resources, LadduError, LadduResult};

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
    /// Merge two transforms, with the left-hand side taking priority where keys overlap.
    ///
    /// # Notes
    /// The receiver (`self`) always wins conflicts; values from `other` only fill in gaps.
    pub fn merged(&self, other: &Self) -> Self {
        let mut merged = self.clone();
        for (key, value) in &other.renames {
            merged
                .renames
                .entry(key.clone())
                .or_insert_with(|| value.clone());
        }
        for (key, value) in &other.fixed {
            merged.fixed.entry(key.clone()).or_insert(*value);
        }
        for name in &other.freed {
            merged.freed.insert(name.clone());
        }
        merged
    }
}

/// Tracks ordered parameter names, their fixed values, and transforms.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ParameterManager {
    ordered_names: Vec<String>,
    name_to_index: HashMap<String, usize>,
    base_fixed: HashMap<String, f64>,
    transform: ParameterTransform,
}

impl ParameterManager {
    /// Build a manager by introspecting [`Resources`].
    pub fn from_resources(resources: &Resources) -> Self {
        let names = resources.parameter_names();
        let fixed_values = resources.fixed_parameter_values();
        let mut manager = Self::with_fixed_values(&names, &fixed_values);
        manager.transform = resources.parameter_overrides.clone();
        manager
    }

    /// Construct a manager containing the provided parameter names as free entries.
    pub fn new_from_names(names: &[String]) -> Self {
        let mut manager = Self::default();
        for name in names {
            manager.append_parameter(name.clone(), None);
        }
        manager
    }

    /// Construct a manager with optional fixed values.
    pub fn with_fixed_values(names: &[String], fixed_values: &HashMap<String, f64>) -> Self {
        let mut manager = Self::default();
        for name in names {
            let fixed = fixed_values.get(name).copied();
            manager.append_parameter(name.clone(), fixed);
        }
        manager
    }

    /// All parameter names in canonical order.
    pub fn parameters(&self) -> Vec<String> {
        self.ordered_names.clone()
    }

    /// Names of the free parameters.
    pub fn free_parameters(&self) -> Vec<String> {
        self.ordered_names
            .iter()
            .filter(|name| self.effective_fixed_value(name).is_none())
            .cloned()
            .collect()
    }

    /// Names of the fixed parameters.
    pub fn fixed_parameters(&self) -> Vec<String> {
        self.ordered_names
            .iter()
            .filter(|name| self.effective_fixed_value(name).is_some())
            .cloned()
            .collect()
    }

    /// Total parameter count.
    pub fn n_parameters(&self) -> usize {
        self.ordered_names.len()
    }

    /// Free parameter count.
    pub fn n_free_parameters(&self) -> usize {
        self.free_parameters().len()
    }

    /// Fixed parameter count.
    pub fn n_fixed_parameters(&self) -> usize {
        self.fixed_parameters().len()
    }

    /// Fixed value for `name`, if any.
    pub fn fixed_value(&self, name: &str) -> Option<f64> {
        self.effective_fixed_value(name)
    }

    /// True if `name` exists in the manager.
    pub fn contains(&self, name: &str) -> bool {
        self.name_to_index.contains_key(name)
    }

    /// Indices (into [`Self::parameters`]) for free parameters.
    pub fn free_parameter_indices(&self) -> Vec<usize> {
        self.ordered_names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| {
                if self.effective_fixed_value(name).is_none() {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Fix `name` to `value`.
    pub fn fix(&self, name: &str, value: f64) -> LadduResult<Self> {
        self.ensure_exists(name)?;
        if let Some(existing) = self.effective_fixed_value(name) {
            if (existing - value).abs() > f64::EPSILON {
                return Err(LadduError::ParameterConflict {
                    name: name.to_string(),
                    reason: "parameter already fixed to a different value".to_string(),
                });
            }
        }
        let mut next = self.clone();
        next.transform.fixed.insert(name.to_string(), value);
        next.transform.freed.shift_remove(name);
        Ok(next)
    }

    /// Free the parameter `name`.
    pub fn free(&self, name: &str) -> LadduResult<Self> {
        self.ensure_exists(name)?;
        let mut next = self.clone();
        next.transform.fixed.remove(name);
        next.transform.freed.insert(name.to_string());
        Ok(next)
    }

    /// Rename `old` to `new`.
    pub fn rename(&self, old: &str, new: &str) -> LadduResult<Self> {
        self.ensure_exists(old)?;
        if self.name_to_index.contains_key(new) && old != new {
            return Err(LadduError::ParameterConflict {
                name: new.to_string(),
                reason: "rename target already exists".to_string(),
            });
        }
        let mut next = self.clone();
        if old != new {
            if let Some(idx) = next.name_to_index.remove(old) {
                next.ordered_names[idx] = new.to_string();
                next.name_to_index.insert(new.to_string(), idx);
            }
            if let Some(value) = next.base_fixed.remove(old) {
                next.base_fixed.insert(new.to_string(), value);
            }
            if let Some(value) = next.transform.fixed.remove(old) {
                next.transform.fixed.insert(new.to_string(), value);
            }
            if next.transform.freed.shift_remove(old) {
                next.transform.freed.insert(new.to_string());
            }
            next.transform
                .renames
                .insert(old.to_string(), new.to_string());
        }
        Ok(next)
    }

    /// Rename parameters according to `mapping`.
    ///
    /// # Notes
    /// Entries are applied sequentially, so later renames can reference the results of earlier
    /// substitutions.
    pub fn rename_parameters(&self, mapping: &HashMap<String, String>) -> LadduResult<Self> {
        let mut next = self.clone();
        for (old, new) in mapping {
            next = next.rename(old, new)?;
        }
        Ok(next)
    }

    /// Assemble the full parameter vector from free values.
    pub fn assemble_full(&self, free_values: &[f64]) -> LadduResult<Vec<f64>> {
        let mut full = Vec::with_capacity(self.ordered_names.len());
        let mut free_iter = free_values.iter();
        for name in &self.ordered_names {
            if let Some(value) = self.effective_fixed_value(name) {
                full.push(value);
            } else if let Some(val) = free_iter.next() {
                full.push(*val);
            } else {
                return Err(LadduError::ParameterConflict {
                    name: "<parameters>".to_string(),
                    reason: format!(
                        "expected {} free values, received {}",
                        self.n_free_parameters(),
                        free_values.len()
                    ),
                });
            }
        }
        if free_iter.next().is_some() {
            return Err(LadduError::ParameterConflict {
                name: "<parameters>".to_string(),
                reason: format!(
                    "expected {} free values, received {}",
                    self.n_free_parameters(),
                    free_values.len()
                ),
            });
        }
        Ok(full)
    }

    /// Merge this manager with `other`, returning the merged order and index maps.
    ///
    /// # Notes
    /// When parameters overlap, the state and value stored in `self` always take precedence over
    /// entries from `other`.
    pub fn merge(&self, other: &Self) -> (Self, Vec<usize>, Vec<usize>) {
        let mut merged = self.clone();
        let left_map: Vec<usize> = (0..self.ordered_names.len()).collect();
        let mut right_map = Vec::with_capacity(other.ordered_names.len());
        for name in &other.ordered_names {
            let fixed = other.effective_fixed_value(name);
            let idx = merged.ensure_parameter(name.clone(), fixed);
            right_map.push(idx);
        }
        (merged, left_map, right_map)
    }

    /// Extend this manager with entries from `other`, returning their indices.
    ///
    /// # Notes
    /// When both managers reference the same parameter, the value and fixed/free status from
    /// `self` are retained.
    pub fn extend_from(&self, other: &Self) -> (Self, Vec<usize>) {
        let mut merged = self.clone();
        let mut indices = Vec::with_capacity(other.ordered_names.len());
        for name in &other.ordered_names {
            let fixed = other.effective_fixed_value(name);
            let idx = merged.ensure_parameter(name.clone(), fixed);
            indices.push(idx);
        }
        (merged, indices)
    }

    /// The accumulated transforms carried by the manager.
    pub fn transform(&self) -> &ParameterTransform {
        &self.transform
    }

    fn ensure_parameter(&mut self, name: String, fixed: Option<f64>) -> usize {
        if let Some(idx) = self.name_to_index.get(&name) {
            if let Some(new_value) = fixed {
                if let Some(existing) = self.effective_fixed_value(&name) {
                    if (existing - new_value).abs() > f64::EPSILON {
                        // Keep existing value from the left-hand set.
                    }
                }
            }
            return *idx;
        }
        let idx = self.ordered_names.len();
        self.append_parameter(name.clone(), fixed);
        idx
    }

    fn append_parameter(&mut self, name: String, fixed: Option<f64>) {
        let idx = self.ordered_names.len();
        self.ordered_names.push(name.clone());
        self.name_to_index.insert(name, idx);
        if let Some(value) = fixed {
            self.base_fixed
                .insert(self.ordered_names[idx].clone(), value);
        }
    }

    fn ensure_exists(&self, name: &str) -> LadduResult<()> {
        if self.name_to_index.contains_key(name) {
            Ok(())
        } else {
            Err(LadduError::UnregisteredParameter {
                name: name.to_string(),
                reason: "parameter not found".to_string(),
            })
        }
    }

    fn effective_fixed_value(&self, name: &str) -> Option<f64> {
        if let Some(value) = self.transform.fixed.get(name) {
            return Some(*value);
        }
        if self.transform.freed.contains(name) {
            return None;
        }
        self.base_fixed.get(name).copied()
    }
}
