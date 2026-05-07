use serde::{Deserialize, Serialize};

use crate::{data::DatasetMetadata, LadduError, LadduResult};

fn names_to_string(names: &[String]) -> String {
    names.join(", ")
}

/// A reusable selection that may span one or more four-momentum names.
///
/// Instances are constructed from metadata-facing identifiers and later bound to
/// column indices so that variable evaluators can resolve aliases or grouped
/// particles efficiently.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct P4Selection {
    names: Vec<String>,
    #[serde(skip, default)]
    indices: Vec<usize>,
}

impl P4Selection {
    pub(crate) fn new_many<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            names: names.into_iter().map(Into::into).collect(),
            indices: Vec::new(),
        }
    }

    pub(crate) fn with_indices<I, S>(names: I, indices: Vec<usize>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            names: names.into_iter().map(Into::into).collect(),
            indices,
        }
    }

    /// Returns the metadata names contributing to this selection.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub(crate) fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let mut resolved = Vec::with_capacity(self.names.len());
        for name in &self.names {
            metadata.append_indices_for_name(name, &mut resolved)?;
        }
        self.indices = resolved;
        Ok(())
    }

    /// The resolved column indices backing this selection.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

/// Helper trait to convert common particle specifications into [`P4Selection`] instances.
pub trait IntoP4Selection {
    /// Convert the input into a [`P4Selection`].
    fn into_selection(self) -> P4Selection;
}

impl IntoP4Selection for P4Selection {
    fn into_selection(self) -> P4Selection {
        self
    }
}

impl IntoP4Selection for &P4Selection {
    fn into_selection(self) -> P4Selection {
        self.clone()
    }
}

impl IntoP4Selection for String {
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(vec![self])
    }
}

impl IntoP4Selection for &String {
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(vec![self.clone()])
    }
}

impl IntoP4Selection for &str {
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(vec![self.to_string()])
    }
}

impl<S> IntoP4Selection for Vec<S>
where
    S: Into<String>,
{
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(self.into_iter().map(Into::into).collect::<Vec<_>>())
    }
}

impl<S> IntoP4Selection for &[S]
where
    S: Clone + Into<String>,
{
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(self.iter().cloned().map(Into::into).collect::<Vec<_>>())
    }
}

impl<S, const N: usize> IntoP4Selection for [S; N]
where
    S: Into<String>,
{
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(self.into_iter().map(Into::into).collect::<Vec<_>>())
    }
}

impl<S, const N: usize> IntoP4Selection for &[S; N]
where
    S: Clone + Into<String>,
{
    fn into_selection(self) -> P4Selection {
        P4Selection::new_many(self.iter().cloned().map(Into::into).collect::<Vec<_>>())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AuxSelection {
    name: String,
    #[serde(skip, default)]
    index: Option<usize>,
}

impl AuxSelection {
    pub(crate) fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            index: None,
        }
    }

    pub(crate) fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let idx = metadata
            .aux_index(&self.name)
            .ok_or_else(|| LadduError::UnknownName {
                category: "aux",
                name: self.name.clone(),
            })?;
        self.index = Some(idx);
        Ok(())
    }

    pub(crate) fn index(&self) -> usize {
        self.index.expect("AuxSelection must be bound before use")
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }
}

pub(crate) fn format_names(names: &[String]) -> String {
    names_to_string(names)
}
