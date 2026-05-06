use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    data::Event,
    expression::{Amplitude, AmplitudeID, Expression, IntoTags, Tags},
    parameters::Parameter,
    resources::{Cache, Parameters, Resources},
    LadduResult, ParameterID, ScalarID,
};

/// A testing [`Amplitude`].
#[derive(Clone, Serialize, Deserialize)]
pub struct TestAmplitude {
    pub(crate) tags: Tags,
    pub(crate) re: Parameter,
    pub(crate) pid_re: ParameterID,
    pub(crate) im: Parameter,
    pub(crate) pid_im: ParameterID,
    pub(crate) beam_energy: ScalarID,
}

impl TestAmplitude {
    /// Create a new testing [`Amplitude`].
    #[allow(clippy::new_ret_no_self)]
    pub fn new(tags: impl IntoTags, re: Parameter, im: Parameter) -> LadduResult<Expression> {
        Self {
            tags: tags.into_tags(),
            re,
            pid_re: Default::default(),
            im,
            pid_im: Default::default(),
            beam_energy: Default::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for TestAmplitude {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_re = resources.register_parameter(&self.re)?;
        self.pid_im = resources.register_parameter(&self.im)?;
        self.beam_energy = resources.register_scalar(None);
        resources.register_amplitude(self.tags.clone())
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        let beam = event.p4_at(0);
        cache.store_scalar(self.beam_energy, beam.e());
    }

    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        Complex64::new(parameters.get(self.pid_re), parameters.get(self.pid_im))
            * cache.get_scalar(self.beam_energy)
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let beam_energy = cache.get_scalar(self.beam_energy);
        if let Some(ind) = parameters.free_index(self.pid_re) {
            gradient[ind] = Complex64::ONE * beam_energy;
        }
        if let Some(ind) = parameters.free_index(self.pid_im) {
            gradient[ind] = Complex64::I * beam_energy;
        }
    }
}
