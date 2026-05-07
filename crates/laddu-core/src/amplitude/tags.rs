use serde::{Deserialize, Serialize};

/// Normalized activation tags for an amplitude use-site.
///
/// Empty strings are ignored, and duplicate tags are removed while preserving the first
/// occurrence. An empty tag set is a normal untagged amplitude use-site; it participates in
/// expression evaluation but cannot be selected by tag activation APIs.
#[derive(Clone, Default, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tags(Vec<String>);

impl Tags {
    /// Construct normalized tags from string-like inputs.
    pub fn new(tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut normalized = Vec::new();
        for tag in tags {
            let tag = tag.into();
            if tag.is_empty() || normalized.contains(&tag) {
                continue;
            }
            normalized.push(tag);
        }
        Self(normalized)
    }

    /// Construct an empty tag set.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Borrow the normalized tags.
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }

    /// Return true if this tag set has no selectable tags.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn display_label(&self) -> String {
        if self.is_empty() {
            "<untagged>".to_string()
        } else {
            self.0.join(",")
        }
    }
}

/// Convert user-facing tag inputs into [`Tags`].
pub trait IntoTags {
    /// Convert into normalized tags.
    fn into_tags(self) -> Tags;
}

impl IntoTags for Tags {
    fn into_tags(self) -> Tags {
        self
    }
}

impl IntoTags for () {
    fn into_tags(self) -> Tags {
        Tags::empty()
    }
}

impl IntoTags for &str {
    fn into_tags(self) -> Tags {
        Tags::new([self])
    }
}

impl IntoTags for String {
    fn into_tags(self) -> Tags {
        Tags::new([self])
    }
}

impl IntoTags for &String {
    fn into_tags(self) -> Tags {
        Tags::new([self.clone()])
    }
}

impl<T: Into<String>> IntoTags for Vec<T> {
    fn into_tags(self) -> Tags {
        Tags::new(self)
    }
}

impl<T: Clone + Into<String>> IntoTags for &[T] {
    fn into_tags(self) -> Tags {
        Tags::new(self.iter().cloned())
    }
}

impl<T: Into<String>, const N: usize> IntoTags for [T; N] {
    fn into_tags(self) -> Tags {
        Tags::new(self)
    }
}
