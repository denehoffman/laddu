#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LikelihoodName(String);

impl LikelihoodName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
