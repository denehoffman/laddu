use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub mod ir;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelName(Arc<str>);

impl KernelName {
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheName(Arc<str>);

impl CacheName {
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelSpec {
    name: KernelName,
}

impl KernelSpec {
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self {
            name: KernelName::new(name),
        }
    }

    pub fn name(&self) -> &KernelName {
        &self.name
    }
}

pub fn kernel(name: impl Into<Arc<str>>) -> KernelSpec {
    KernelSpec::new(name)
}
