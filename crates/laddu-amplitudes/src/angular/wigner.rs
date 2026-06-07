use laddu_core::{
    amplitude::{
        debug_key, Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression, IntoTags, Tags,
    },
    data::{DatasetMetadata, Event},
    math::WignerDMatrix,
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    traits::Variable,
    variables::Angles,
    LadduResult, SpinState, J, M,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

/// An amplitude evaluating a Wigner-D matrix element from decay angles.
#[derive(Clone, Serialize, Deserialize)]
pub struct WignerD {
    tags: Tags,
    spin: J,
    row_projection: M,
    column_projection: M,
    dmatrix: WignerDMatrix,
    costheta: Box<dyn Variable>,
    phi: Box<dyn Variable>,
    angles_key: String,
    value_id: ComplexScalarID,
}

impl WignerD {
    /// Construct a new Wigner-D amplitude.
    ///
    /// The returned expression evaluates
    /// `D^j_{m' m}(phi, theta, 0)`, with `theta = acos(costheta)`.
    pub fn new(
        tags: impl IntoTags,
        spin: J,
        row_projection: M,
        column_projection: M,
        angles: &Angles,
    ) -> LadduResult<Expression> {
        SpinState::new(spin, row_projection)?;
        SpinState::new(spin, column_projection)?;
        let dmatrix = WignerDMatrix::new(spin, row_projection, column_projection)?;
        Self {
            tags: tags.into_tags(),
            spin,
            row_projection,
            column_projection,
            dmatrix,
            costheta: angles.costheta_variable(),
            phi: angles.phi_variable(),
            angles_key: angles.to_string(),
            value_id: ComplexScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for WignerD {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.value_id = resources.register_complex_scalar(None);
        resources.register_amplitude(self.tags.clone())
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("WignerD")
                .with_field("spin", self.spin.doubled().to_string())
                .with_field("row_projection", self.row_projection.doubled().to_string())
                .with_field(
                    "column_projection",
                    self.column_projection.doubled().to_string(),
                )
                .with_field("angles", debug_key(&self.angles_key)),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.costheta.bind(metadata)?;
        self.phi.bind(metadata)
    }

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        let costheta = self.costheta.value(event).clamp(-1.0, 1.0);
        let theta = costheta.acos();
        let phi = self.phi.value(event);
        cache.store_complex_scalar(self.value_id, self.dmatrix.D(phi, theta, 0.0));
    }

    fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        cache.get_complex_scalar(self.value_id)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
    }
}
