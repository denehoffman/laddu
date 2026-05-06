use std::{array, collections::HashMap};

use serde::{Deserialize, Serialize};

use super::{
    cache::Cache,
    cache_ids::{ComplexMatrixID, ComplexScalarID, ComplexVectorID, MatrixID, ScalarID, VectorID},
    parameter_store::ParameterID,
};
use crate::{
    amplitudes::{AmplitudeID, IntoTags, Parameter, ParameterMap, Tags},
    LadduError, LadduResult,
};

fn is_glob_selector(selector: &str) -> bool {
    selector.contains('*') || selector.contains('?')
}

fn glob_matches(pattern: &str, text: &str) -> bool {
    let pattern = pattern.chars().collect::<Vec<_>>();
    let text = text.chars().collect::<Vec<_>>();
    let mut table = vec![vec![false; text.len() + 1]; pattern.len() + 1];
    table[0][0] = true;
    for pattern_idx in 1..=pattern.len() {
        if pattern[pattern_idx - 1] == '*' {
            table[pattern_idx][0] = table[pattern_idx - 1][0];
        }
    }
    for pattern_idx in 1..=pattern.len() {
        for text_idx in 1..=text.len() {
            table[pattern_idx][text_idx] = match pattern[pattern_idx - 1] {
                '*' => table[pattern_idx - 1][text_idx] || table[pattern_idx][text_idx - 1],
                '?' => table[pattern_idx - 1][text_idx - 1],
                character => {
                    character == text[text_idx - 1] && table[pattern_idx - 1][text_idx - 1]
                }
            };
        }
    }
    table[pattern.len()][text.len()]
}

/// The main resource manager for cached values, amplitudes, parameters, and constants.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Resources {
    amplitudes: HashMap<String, Vec<usize>>,
    untagged_amplitudes: Vec<bool>,
    /// A list indicating which amplitude use-sites are active.
    pub active: Vec<bool>,
    #[serde(default)]
    active_indices: Vec<usize>,
    /// The registered parameters and constants used by this resource set.
    pub parameter_map: ParameterMap,
    /// The [`Cache`] for each [`EventData`](crate::data::EventData)
    pub caches: Vec<Cache>,
    scalar_cache_names: HashMap<String, usize>,
    complex_scalar_cache_names: HashMap<String, usize>,
    vector_cache_names: HashMap<String, usize>,
    complex_vector_cache_names: HashMap<String, usize>,
    matrix_cache_names: HashMap<String, usize>,
    complex_matrix_cache_names: HashMap<String, usize>,
    cache_size: usize,
}

impl Resources {
    /// Rename a single registered parameter.
    pub fn rename_parameter(&mut self, old: &str, new: &str) -> LadduResult<()> {
        self.parameter_map.rename_parameter(old, new)
    }

    /// Rename multiple registered parameters.
    pub fn rename_parameters(&mut self, mapping: &HashMap<String, String>) -> LadduResult<()> {
        self.parameter_map.rename_parameters(mapping)
    }

    /// Mark a registered parameter as free.
    pub fn free_parameter(&self, name: &str) -> LadduResult<()> {
        self.parameter_map.free_parameter(name)
    }

    /// Fix a registered parameter to the supplied value.
    pub fn fix_parameter(&self, name: &str, value: f64) -> LadduResult<()> {
        self.parameter_map.fix_parameter(name, value)
    }

    /// The list of free parameter names.
    pub fn free_parameter_names(&self) -> Vec<String> {
        self.parameter_map.free().names()
    }

    /// The list of fixed parameter names.
    pub fn fixed_parameter_names(&self) -> Vec<String> {
        self.parameter_map.fixed().names()
    }

    /// All parameter names (free first, then fixed).
    pub fn parameter_names(&self) -> Vec<String> {
        self.free_parameter_names()
            .into_iter()
            .chain(self.fixed_parameter_names())
            .collect()
    }

    /// The registered parameters.
    pub fn parameters(&self) -> ParameterMap {
        self.parameter_map.clone()
    }

    /// Number of free parameters.
    pub fn n_free_parameters(&self) -> usize {
        self.parameter_map.free().len()
    }

    /// Number of fixed parameters.
    pub fn n_fixed_parameters(&self) -> usize {
        self.parameter_map.fixed().len()
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

    fn selector_indices(&self, selector: &str) -> Vec<usize> {
        if is_glob_selector(selector) {
            self.amplitudes
                .iter()
                .filter_map(|(tag, amplitudes)| {
                    if glob_matches(selector, tag) {
                        Some(amplitudes.iter().copied())
                    } else {
                        None
                    }
                })
                .flatten()
                .collect()
        } else {
            self.amplitudes.get(selector).cloned().unwrap_or_default()
        }
    }

    fn set_activation_state_by_selector(
        &mut self,
        selector: &str,
        active: bool,
        strict: bool,
    ) -> LadduResult<bool> {
        let indices = self.selector_indices(selector);
        if indices.is_empty() {
            if strict {
                return Err(LadduError::AmplitudeNotFoundError {
                    name: selector.to_string(),
                });
            }
            return Ok(false);
        }
        let mut changed = false;
        for idx in indices {
            if self.untagged_amplitudes.get(idx).copied().unwrap_or(false) {
                continue;
            }
            if self.active[idx] != active {
                self.active[idx] = active;
                changed = true;
            }
        }
        Ok(changed)
    }

    fn selector_indices_many<T: AsRef<str>>(
        &self,
        selectors: &[T],
        strict: bool,
    ) -> LadduResult<Vec<usize>> {
        let mut indices = Vec::new();
        for selector in selectors {
            let selector_ref = selector.as_ref();
            let selector_indices = self.selector_indices(selector_ref);
            if selector_indices.is_empty() && strict {
                return Err(LadduError::AmplitudeNotFoundError {
                    name: selector_ref.to_string(),
                });
            }
            indices.extend(selector_indices);
        }
        indices.sort_unstable();
        indices.dedup();
        Ok(indices)
    }

    fn isolate_indices(&mut self, indices: &[usize]) {
        let mut changed = false;
        for (idx, active) in self.active.iter_mut().enumerate() {
            let next_active = self.untagged_amplitudes.get(idx).copied().unwrap_or(false)
                || indices.binary_search(&idx).is_ok();
            if *active != next_active {
                *active = next_active;
                changed = true;
            }
        }
        if changed {
            self.rebuild_active_indices();
        }
    }

    /// Activate [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector.
    pub fn activate<T: AsRef<str>>(&mut self, name: T) {
        if self
            .set_activation_state_by_selector(name.as_ref(), true, false)
            .unwrap_or(false)
        {
            self.rebuild_active_indices();
        }
    }
    /// Activate several [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector.
    pub fn activate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        let mut changed = false;
        for name in names {
            if self
                .set_activation_state_by_selector(name.as_ref(), true, false)
                .unwrap_or(false)
            {
                changed = true;
            }
        }
        if changed {
            self.rebuild_active_indices();
        }
    }
    /// Activate [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector, returning an error if no use-site matches.
    pub fn activate_strict<T: AsRef<str>>(&mut self, name: T) -> LadduResult<()> {
        if self.set_activation_state_by_selector(name.as_ref(), true, true)? {
            self.rebuild_active_indices();
        }
        Ok(())
    }
    /// Activate several [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector, returning an error if any selector has no matches.
    pub fn activate_many_strict<T: AsRef<str>>(&mut self, names: &[T]) -> LadduResult<()> {
        let mut changed = false;
        for name in names {
            if self.set_activation_state_by_selector(name.as_ref(), true, true)? {
                changed = true;
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
    /// Deactivate [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector.
    pub fn deactivate<T: AsRef<str>>(&mut self, name: T) {
        if self
            .set_activation_state_by_selector(name.as_ref(), false, false)
            .unwrap_or(false)
        {
            self.rebuild_active_indices();
        }
    }
    /// Deactivate several [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector.
    pub fn deactivate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        let mut changed = false;
        for name in names {
            if self
                .set_activation_state_by_selector(name.as_ref(), false, false)
                .unwrap_or(false)
            {
                changed = true;
            }
        }
        if changed {
            self.rebuild_active_indices();
        }
    }
    /// Deactivate [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector, returning an error if no use-site matches.
    pub fn deactivate_strict<T: AsRef<str>>(&mut self, name: T) -> LadduResult<()> {
        if self.set_activation_state_by_selector(name.as_ref(), false, true)? {
            self.rebuild_active_indices();
        }
        Ok(())
    }
    /// Deactivate several [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector, returning an error if any selector has no matches.
    pub fn deactivate_many_strict<T: AsRef<str>>(&mut self, names: &[T]) -> LadduResult<()> {
        let mut changed = false;
        for name in names {
            if self.set_activation_state_by_selector(name.as_ref(), false, true)? {
                changed = true;
            }
        }
        if changed {
            self.rebuild_active_indices();
        }
        Ok(())
    }
    /// Deactivate all tagged [`Amplitude`](crate::amplitudes::Amplitude) use-sites.
    pub fn deactivate_all(&mut self) {
        let mut changed = false;
        for (idx, active) in self.active.iter_mut().enumerate() {
            if self.untagged_amplitudes.get(idx).copied().unwrap_or(false) {
                continue;
            }
            if *active {
                *active = false;
                changed = true;
            }
        }
        if changed {
            self.rebuild_active_indices();
        }
    }
    /// Isolate [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector.
    pub fn isolate<T: AsRef<str>>(&mut self, name: T) {
        let indices = self.selector_indices(name.as_ref());
        if !indices.is_empty() {
            self.isolate_indices(&indices);
        }
    }
    /// Isolate [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector, returning an error if no use-site matches.
    pub fn isolate_strict<T: AsRef<str>>(&mut self, name: T) -> LadduResult<()> {
        let indices = self.selector_indices_many(&[name], true)?;
        self.isolate_indices(&indices);
        Ok(())
    }
    /// Isolate several [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector.
    pub fn isolate_many<T: AsRef<str>>(&mut self, names: &[T]) {
        if let Ok(indices) = self.selector_indices_many(names, false) {
            self.isolate_indices(&indices);
        }
    }
    /// Isolate several [`Amplitude`](crate::amplitudes::Amplitude) use-sites by tag or glob selector, returning an error if any selector has no matches.
    pub fn isolate_many_strict<T: AsRef<str>>(&mut self, names: &[T]) -> LadduResult<()> {
        let indices = self.selector_indices_many(names, true)?;
        self.isolate_indices(&indices);
        Ok(())
    }
    /// Register an [`Amplitude`](crate::amplitudes::Amplitude) use-site with activation tags.
    /// This method should be called at the end of the
    /// [`Amplitude::register`](crate::amplitudes::Amplitude::register) method. The
    /// The amplitude should store normalized activation tags and pass them here.
    ///
    /// # Errors
    ///
    /// Empty tag strings are discarded. If no non-empty tags remain, the amplitude use-site is
    /// untagged; it still participates in evaluation but cannot be selected by
    /// activation/deactivation/isolation APIs.
    pub fn register_amplitude(&mut self, tags: impl IntoTags) -> LadduResult<AmplitudeID> {
        let tags = tags.into_tags();
        let next_id = AmplitudeID(tags.clone(), self.active.len());
        for tag in tags.as_slice() {
            self.amplitudes
                .entry(tag.clone())
                .or_default()
                .push(next_id.1);
        }
        self.active.push(true);
        self.untagged_amplitudes.push(tags.is_empty());
        self.rebuild_active_indices();
        Ok(next_id)
    }

    /// Fetch the first [`AmplitudeID`] for a previously registered amplitude by tag.
    pub fn amplitude_id(&self, tag: &str) -> Option<AmplitudeID> {
        self.amplitudes
            .get(tag)
            .and_then(|indices| indices.first().copied())
            .map(|idx| AmplitudeID(Tags::new([tag]), idx))
    }

    pub(crate) fn configure_amplitude_tags(&mut self, tags: &[Tags]) {
        self.amplitudes.clear();
        self.active = vec![true; tags.len()];
        self.untagged_amplitudes = tags.iter().map(Tags::is_empty).collect();
        for (idx, amplitude_tags) in tags.iter().enumerate() {
            for tag in amplitude_tags.as_slice() {
                self.amplitudes.entry(tag.clone()).or_default().push(idx);
            }
        }
        self.rebuild_active_indices();
    }

    pub(crate) fn apply_active_mask(&mut self, mask: &[bool]) -> LadduResult<()> {
        for (idx, &active) in mask.iter().enumerate() {
            if !active && self.untagged_amplitudes.get(idx).copied().unwrap_or(false) {
                return Err(LadduError::Custom(
                    "active mask cannot deactivate untagged amplitudes".to_string(),
                ));
            }
        }
        self.active.clone_from_slice(mask);
        self.rebuild_active_indices();
        Ok(())
    }

    /// Register a parameter. This method should be called within
    /// [`Amplitude::register`](crate::amplitudes::Amplitude::register).
    /// The resulting [`ParameterID`] should be stored to retrieve the value from the
    /// [`Parameters`](crate::resources::Parameters) wrapper.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter is unnamed, if the name is reused with incompatible
    /// fixed/free status or fixed value, or if renaming causes a conflict.
    pub fn register_parameter(&mut self, p: &Parameter) -> LadduResult<ParameterID> {
        self.parameter_map.register_parameter(p)
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
