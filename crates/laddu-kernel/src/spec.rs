use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Stable user-facing name of a computational kernel.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelName(Arc<str>);

impl KernelName {
    /// Creates a kernel name.
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self(name.into())
    }

    /// Returns the name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Stable user-facing name of a cached event quantity.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheName(Arc<str>);

impl CacheName {
    /// Creates a cache name.
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self(name.into())
    }

    /// Returns the name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Declarative specification for a named kernel.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelSpec {
    name: KernelName,
}

impl KernelSpec {
    /// Creates a named kernel specification.
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self {
            name: KernelName::new(name),
        }
    }

    /// Returns the kernel name.
    pub fn name(&self) -> &KernelName {
        &self.name
    }
}

/// Creates a named [`KernelSpec`].
pub fn kernel(name: impl Into<Arc<str>>) -> KernelSpec {
    KernelSpec::new(name)
}
