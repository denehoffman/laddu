use indexmap::IndexMap;

use crate::{variables::P4Selection, LadduError, LadduResult};

/// A collection of [`EventData`](crate::data::EventData).
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub(crate) p4_names: Vec<String>,
    pub(crate) aux_names: Vec<String>,
    pub(crate) p4_lookup: IndexMap<String, usize>,
    pub(crate) aux_lookup: IndexMap<String, usize>,
    pub(crate) p4_selections: IndexMap<String, P4Selection>,
}

impl DatasetMetadata {
    /// Construct metadata from explicit particle and auxiliary names.
    pub fn new<P: Into<String>, A: Into<String>>(
        p4_names: Vec<P>,
        aux_names: Vec<A>,
    ) -> LadduResult<Self> {
        let mut p4_lookup = IndexMap::with_capacity(p4_names.len());
        let mut aux_lookup = IndexMap::with_capacity(aux_names.len());
        let mut p4_selections = IndexMap::with_capacity(p4_names.len());
        let p4_names: Vec<String> = p4_names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| {
                let name = name.into();
                if p4_lookup.contains_key(&name) {
                    return Err(LadduError::DuplicateName {
                        category: "p4",
                        name,
                    });
                }
                p4_lookup.insert(name.clone(), idx);
                p4_selections.insert(
                    name.clone(),
                    P4Selection::with_indices(vec![name.clone()], vec![idx]),
                );
                Ok(name)
            })
            .collect::<Result<_, _>>()?;
        let aux_names: Vec<String> = aux_names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| {
                let name = name.into();
                if aux_lookup.contains_key(&name) {
                    return Err(LadduError::DuplicateName {
                        category: "aux",
                        name,
                    });
                }
                aux_lookup.insert(name.clone(), idx);
                Ok(name)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            p4_names,
            aux_names,
            p4_lookup,
            aux_lookup,
            p4_selections,
        })
    }

    /// Create metadata with no registered names.
    pub fn empty() -> Self {
        Self {
            p4_names: Vec::new(),
            aux_names: Vec::new(),
            p4_lookup: IndexMap::new(),
            aux_lookup: IndexMap::new(),
            p4_selections: IndexMap::new(),
        }
    }

    /// Resolve the index of a four-momentum by name.
    pub fn p4_index(&self, name: &str) -> Option<usize> {
        self.p4_lookup.get(name).copied()
    }

    /// Registered four-momentum names in declaration order.
    pub fn p4_names(&self) -> &[String] {
        &self.p4_names
    }

    /// Resolve the index of an auxiliary scalar by name.
    pub fn aux_index(&self, name: &str) -> Option<usize> {
        self.aux_lookup.get(name).copied()
    }

    /// Registered auxiliary scalar names in declaration order.
    pub fn aux_names(&self) -> &[String] {
        &self.aux_names
    }

    pub(crate) fn ensure_new_p4_name(&self, name: &str) -> LadduResult<()> {
        if self.p4_selections.contains_key(name) {
            return Err(LadduError::DuplicateName {
                category: "p4",
                name: name.to_string(),
            });
        }
        Ok(())
    }

    pub(crate) fn ensure_new_aux_name(&self, name: &str) -> LadduResult<()> {
        if self.aux_lookup.contains_key(name) {
            return Err(LadduError::DuplicateName {
                category: "aux",
                name: name.to_string(),
            });
        }
        Ok(())
    }

    pub(crate) fn add_p4_name<N>(&mut self, name: N) -> LadduResult<()>
    where
        N: Into<String>,
    {
        let name = name.into();
        self.ensure_new_p4_name(&name)?;
        let index = self.p4_names.len();
        self.p4_lookup.insert(name.clone(), index);
        self.p4_selections.insert(
            name.clone(),
            P4Selection::with_indices(vec![name.clone()], vec![index]),
        );
        self.p4_names.push(name);
        Ok(())
    }

    pub(crate) fn add_aux_name<N>(&mut self, name: N) -> LadduResult<()>
    where
        N: Into<String>,
    {
        let name = name.into();
        self.ensure_new_aux_name(&name)?;
        let index = self.aux_names.len();
        self.aux_lookup.insert(name.clone(), index);
        self.aux_names.push(name);
        Ok(())
    }

    /// Look up a resolved four-momentum selection by name (canonical or alias).
    pub fn p4_selection(&self, name: &str) -> Option<&P4Selection> {
        self.p4_selections.get(name)
    }

    /// Register an alias mapping to one or more existing four-momenta.
    pub fn add_p4_alias<N>(&mut self, alias: N, mut selection: P4Selection) -> LadduResult<()>
    where
        N: Into<String>,
    {
        let alias = alias.into();
        if self.p4_selections.contains_key(&alias) {
            return Err(LadduError::DuplicateName {
                category: "alias",
                name: alias,
            });
        }
        selection.bind(self)?;
        self.p4_selections.insert(alias, selection);
        Ok(())
    }

    /// Register multiple aliases at once.
    pub fn add_p4_aliases<I, N>(&mut self, entries: I) -> LadduResult<()>
    where
        I: IntoIterator<Item = (N, P4Selection)>,
        N: Into<String>,
    {
        for (alias, selection) in entries {
            self.add_p4_alias(alias, selection)?;
        }
        Ok(())
    }

    pub(crate) fn append_indices_for_name(
        &self,
        name: &str,
        target: &mut Vec<usize>,
    ) -> LadduResult<()> {
        if let Some(selection) = self.p4_selections.get(name) {
            target.extend_from_slice(selection.indices());
            return Ok(());
        }
        Err(LadduError::UnknownName {
            category: "p4",
            name: name.to_string(),
        })
    }
}

impl Default for DatasetMetadata {
    fn default() -> Self {
        Self::empty()
    }
}
