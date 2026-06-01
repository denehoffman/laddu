use serde::{Deserialize, Serialize};

use super::support::unit_vector;
use crate::{vectors::Vec3, LadduError, LadduResult};

/// A symbolic angular frame definition.
///
/// A [`Frame`] declares the vertex rest frame where a momentum is measured and the symbolic axes
/// used to project that momentum. Event-specific numeric axes are represented by [`FrameAxes`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Frame {
    origin: String,
    axes: Axes,
}

impl Frame {
    /// Construct a symbolic frame at `origin` from symbolic axes.
    pub fn new(origin: impl Into<String>, axes: Axes) -> LadduResult<Self> {
        let origin = origin.into();
        if origin.trim().is_empty() {
            return Err(LadduError::Custom(
                "frame origin cannot be empty".to_string(),
            ));
        }
        Ok(Self { origin, axes })
    }

    /// Return the vertex where measured momenta are projected.
    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// Return the symbolic axes for this frame.
    pub fn axes(&self) -> &Axes {
        &self.axes
    }
}

/// Symbolic axis definitions used to construct a frame basis.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Axes {
    y: Axis,
    z: Axis,
}

impl Axes {
    /// Construct axes from a `y` axis recipe and a `z` axis recipe.
    pub fn from_y_z(y: Axis, z: Axis) -> Self {
        Self { y, z }
    }

    /// Return the symbolic `y` axis recipe.
    pub fn y(&self) -> &Axis {
        &self.y
    }

    /// Return the symbolic `z` axis recipe.
    pub fn z(&self) -> &Axis {
        &self.z
    }
}

/// A symbolic axis recipe.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Axis {
    source: AxisSource,
    at: String,
    sign: AxisSign,
}

impl Axis {
    /// Construct an axis along a particle momentum.
    pub fn particle(particle: impl Into<String>) -> Self {
        Self {
            source: AxisSource::Particle(particle.into()),
            at: String::new(),
            sign: AxisSign::Along,
        }
    }

    /// Construct an axis opposite a particle momentum.
    pub fn opposite(particle: impl Into<String>) -> Self {
        Self::particle(particle).flipped()
    }

    /// Construct an axis normal to the plane spanned by two particle momenta.
    ///
    /// The primitive convention is `a x b`. Use [`Axis::flipped`] to reverse this orientation.
    pub fn normal(a: impl Into<String>, b: impl Into<String>) -> Self {
        Self {
            source: AxisSource::Normal {
                a: a.into(),
                b: b.into(),
            },
            at: String::new(),
            sign: AxisSign::Along,
        }
    }

    /// Set the vertex frame where this axis source is evaluated.
    pub fn at(mut self, vertex: impl Into<String>) -> Self {
        self.at = vertex.into();
        self
    }

    /// Return a copy with the axis orientation reversed.
    pub fn flipped(mut self) -> Self {
        self.sign = self.sign.flipped();
        self
    }

    /// Return the axis source.
    pub fn source(&self) -> &AxisSource {
        &self.source
    }

    /// Return the vertex frame where this axis is evaluated.
    pub fn frame(&self) -> &str {
        &self.at
    }

    /// Return the axis orientation sign.
    pub fn sign(&self) -> AxisSign {
        self.sign
    }
}

/// The source of a symbolic axis.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AxisSource {
    /// A single particle momentum.
    Particle(String),
    /// The cross product of two particle momenta.
    Normal {
        /// First particle in `a x b`.
        a: String,
        /// Second particle in `a x b`.
        b: String,
    },
}

/// Whether a symbolic axis points along or opposite its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AxisSign {
    /// Point along the source direction.
    Along,
    /// Point opposite the source direction.
    Opposite,
}

impl AxisSign {
    fn flipped(self) -> Self {
        match self {
            Self::Along => Self::Opposite,
            Self::Opposite => Self::Along,
        }
    }

    pub(crate) fn apply(self, vector: Vec3) -> Vec3 {
        match self {
            Self::Along => vector,
            Self::Opposite => -vector,
        }
    }
}

/// Orthonormal axes used to project decay momenta into an angular-analysis frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrameAxes {
    x: Vec3,
    y: Vec3,
    z: Vec3,
}

impl Default for FrameAxes {
    fn default() -> Self {
        Self {
            x: Vec3::x(),
            y: Vec3::y(),
            z: Vec3::z(),
        }
    }
}

impl FrameAxes {
    /// Construct normalized right-handed axes after validating orthogonality.
    pub fn new(x: Vec3, y: Vec3, z: Vec3) -> LadduResult<Self> {
        const ORTHOGONALITY_TOL: f64 = 1.0e-12;
        const HANDEDNESS_TOL: f64 = 1.0e-12;

        let x = unit_vector(x, "x axis")?;
        let y = unit_vector(y, "y axis")?;
        let z = unit_vector(z, "z axis")?;
        if x.dot(&y).abs() > ORTHOGONALITY_TOL
            || x.dot(&z).abs() > ORTHOGONALITY_TOL
            || y.dot(&z).abs() > ORTHOGONALITY_TOL
        {
            return Err(LadduError::Custom(
                "frame axes must be mutually orthogonal".to_string(),
            ));
        }
        if x.cross(&y).dot(&z) < -HANDEDNESS_TOL {
            return Err(LadduError::Custom(
                "frame axes must form a right-handed basis".to_string(),
            ));
        }
        Ok(Self { x, y, z })
    }

    /// Construct right-handed axes from a `z` axis and a plane normal.
    pub fn from_y_z(y: Vec3, z: Vec3) -> LadduResult<Self> {
        let z = unit_vector(z, "z axis")?;
        let y = unit_vector(y, "y axis")?;
        let x = unit_vector(y.cross(&z), "x axis")?;
        Self::new(x, y, z)
    }

    /// Construct daughter axes in the current rest frame.
    ///
    /// The daughter direction defines the new `z` axis. The current `z` axis and daughter
    /// direction define the rotation plane, which makes this helper reusable with any parent-axis
    /// convention.
    pub fn for_daughter(self, daughter_momentum: Vec3) -> LadduResult<Self> {
        let z = unit_vector(daughter_momentum, "daughter z axis")?;
        let mut plane_normal = self.z.cross(&z);
        if plane_normal.mag2() <= f64::EPSILON * f64::EPSILON {
            plane_normal = self.y;
        }
        Self::from_y_z(plane_normal, z)
    }

    /// Return the unit `x` axis.
    pub const fn x(&self) -> Vec3 {
        self.x
    }

    /// Return the unit `y` axis.
    pub const fn y(&self) -> Vec3 {
        self.y
    }

    /// Return the unit `z` axis.
    pub const fn z(&self) -> Vec3 {
        self.z
    }

    /// Project a vector onto these frame axes.
    pub fn components(&self, vector: &Vec3) -> Vec3 {
        Vec3::new(
            vector.dot(&self.x),
            vector.dot(&self.y),
            vector.dot(&self.z),
        )
    }

    /// Return the cosine of the polar angle of `vector` in these axes.
    pub fn costheta(&self, vector: &Vec3) -> f64 {
        self.components(vector).costheta()
    }

    /// Return the azimuthal angle of `vector` in these axes.
    pub fn phi(&self, vector: &Vec3) -> f64 {
        self.components(vector).phi()
    }

    /// Return the polar angle of `vector` in these axes.
    pub fn theta(&self, vector: &Vec3) -> f64 {
        self.costheta(vector).acos()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn assert_vec3_close(actual: Vec3, expected: Vec3) {
        assert_relative_eq!(actual.x, expected.x);
        assert_relative_eq!(actual.y, expected.y);
        assert_relative_eq!(actual.z, expected.z);
    }

    #[test]
    fn symbolic_frame_keeps_axis_recipes() {
        let frame = Frame::new(
            "rho_decay",
            Axes::from_y_z(
                Axis::normal("beam", "spectator").at("production").flipped(),
                Axis::opposite("spectator").at("rho_decay"),
            ),
        )
        .unwrap();

        assert_eq!(frame.origin(), "rho_decay");
        assert_eq!(frame.axes().y().frame(), "production");
        assert_eq!(frame.axes().z().frame(), "rho_decay");
        assert_eq!(frame.axes().y().sign(), AxisSign::Opposite);
    }

    #[test]
    fn daughter_axes_support_cascade_frames() {
        let child_axes = FrameAxes::default()
            .for_daughter(Vec3::new(1.0, 0.0, 0.0))
            .unwrap();

        assert_vec3_close(child_axes.x(), Vec3::new(0.0, 0.0, -1.0));
        assert_vec3_close(child_axes.y(), Vec3::new(0.0, 1.0, 0.0));
        assert_vec3_close(child_axes.z(), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn projected_decay_angles_pin_azimuth_sign() {
        let axes = FrameAxes::from_y_z(Vec3::y(), Vec3::x()).unwrap();
        let costheta = axes.costheta(&Vec3::new(0.0, 0.0, 1.0));
        let phi = axes.phi(&Vec3::new(0.0, 0.0, 1.0));

        assert_relative_eq!(costheta, 0.0);
        assert_relative_eq!(phi, std::f64::consts::PI);
    }

    #[test]
    fn frame_axes_reject_degenerate_plane() {
        let err = FrameAxes::from_y_z(Vec3::zero(), Vec3::z());

        assert!(err.is_err());
    }
}
