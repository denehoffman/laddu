use laddu_compile::CompiledModel;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AutodiffMode {
    Forward,
    Reverse,
}

#[derive(Clone, Debug)]
pub struct AutodiffPlan {
    mode: AutodiffMode,
    parameter_count: usize,
}

impl AutodiffPlan {
    pub fn from_model(model: &CompiledModel, mode: AutodiffMode) -> Self {
        Self {
            mode,
            parameter_count: model.params().n_free(),
        }
    }

    pub fn mode(&self) -> AutodiffMode {
        self.mode
    }

    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }
}
