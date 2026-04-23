use serde::{Deserialize, Serialize};

use super::{
    decay_angles::DecayAngles,
    rest_frame::RestFrame,
    support::{checked_boost_vector, unit_vector},
};
use crate::{
    quantum::Frame,
    vectors::{Vec3, Vec4},
    LadduError, LadduResult,
};

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
    pub fn from_z_and_plane_normal(z: Vec3, plane_normal: Vec3) -> LadduResult<Self> {
        let z = unit_vector(z, "z axis")?;
        let plane_normal = unit_vector(plane_normal, "frame-plane normal")?;
        let x = unit_vector(plane_normal.cross(&z), "x axis")?;
        let y = unit_vector(z.cross(&x), "y axis")?;
        Self::new(x, y, z)
    }

    /// Construct production-frame axes from caller-selected production momenta.
    ///
    /// `reference`, `parent`, and `spectator` are lab-frame four-momenta. `system_boost` is the
    /// boost into the frame where the production plane is defined. This keeps named event topology
    /// handling outside the frame helper while still sharing the convention-sensitive geometry.
    pub fn from_production_frame(
        frame: Frame,
        reference: Vec4,
        parent: Vec4,
        spectator: Vec4,
        system_boost: Vec3,
    ) -> LadduResult<Self> {
        checked_boost_vector(system_boost, "production system")?;
        let reference = reference.boost(&system_boost);
        let parent = parent.boost(&system_boost);
        let spectator = spectator.boost(&system_boost);

        let parent_rest = RestFrame::new(parent)?;
        let reference_in_parent = parent_rest.transform(reference).vec3();
        let spectator_in_parent = parent_rest.transform(spectator).vec3();

        let plane_normal = unit_vector(
            reference_in_parent.cross(&(-spectator_in_parent)),
            "production-plane normal",
        )?;

        let z = match frame {
            Frame::Helicity => unit_vector(-spectator_in_parent, "production-frame z axis")?,
            Frame::GottfriedJackson => {
                unit_vector(reference_in_parent, "Gottfried-Jackson z axis")?
            }
            Frame::Adair => {
                return Err(LadduError::Custom(
                    "Adair frame construction is not implemented yet".to_string(),
                ));
            }
        };

        Self::from_z_and_plane_normal(z, plane_normal)
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
        Self::from_z_and_plane_normal(z, plane_normal)
    }

    /// Return the unit `x` axis.
    pub const fn x(self) -> Vec3 {
        self.x
    }

    /// Return the unit `y` axis.
    pub const fn y(self) -> Vec3 {
        self.y
    }

    /// Return the unit `z` axis.
    pub const fn z(self) -> Vec3 {
        self.z
    }

    /// Project a vector onto these frame axes.
    pub fn components(self, vector: Vec3) -> Vec3 {
        Vec3::new(
            vector.dot(&self.x),
            vector.dot(&self.y),
            vector.dot(&self.z),
        )
    }

    /// Compute spherical decay angles for a vector expressed in the same rest frame as the axes.
    pub fn angles(self, vector: Vec3) -> LadduResult<DecayAngles> {
        let components = self.components(vector);
        DecayAngles::from_components(components)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn transverse_momenta() -> (Vec4, Vec4, Vec4) {
        (
            Vec4::new(0.0, 0.0, 5.0, 5.0),
            Vec4::new(1.0, 0.0, 0.0, 2.0),
            Vec4::new(-1.0, 0.0, 0.0, 2.0),
        )
    }

    fn transverse_axes(frame: Frame) -> FrameAxes {
        let (reference, parent, spectator) = transverse_momenta();
        FrameAxes::from_production_frame(frame, reference, parent, spectator, Vec3::zero()).unwrap()
    }

    fn assert_vec3_close(actual: Vec3, expected: Vec3) {
        assert_relative_eq!(actual.x, expected.x);
        assert_relative_eq!(actual.y, expected.y);
        assert_relative_eq!(actual.z, expected.z);
    }

    #[test]
    fn helicity_frame_axes_are_fixed_for_transverse_production() {
        let axes = transverse_axes(Frame::Helicity);

        assert_vec3_close(axes.x(), Vec3::new(0.0, 0.0, -1.0));
        assert_vec3_close(axes.y(), Vec3::new(0.0, 1.0, 0.0));
        assert_vec3_close(axes.z(), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn gottfried_jackson_frame_uses_boosted_reference_axis() {
        let axes = transverse_axes(Frame::GottfriedJackson);
        let sqrt_three_over_two = 3.0_f64.sqrt() / 2.0;

        assert_vec3_close(axes.x(), Vec3::new(sqrt_three_over_two, 0.0, 0.5));
        assert_vec3_close(axes.y(), Vec3::new(0.0, 1.0, 0.0));
        assert_vec3_close(axes.z(), Vec3::new(-0.5, 0.0, sqrt_three_over_two));
    }

    #[test]
    fn rest_frame_boosts_parent_to_zero_momentum() {
        let parent = Vec4::new(1.0, 2.0, 3.0, 5.0);
        let rest_parent = RestFrame::new(parent).unwrap().transform(parent);

        assert_relative_eq!(rest_parent.px(), 0.0);
        assert_relative_eq!(rest_parent.py(), 0.0);
        assert_relative_eq!(rest_parent.pz(), 0.0);
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
        let axes = transverse_axes(Frame::Helicity);
        let angles = axes.angles(Vec3::new(0.0, 0.0, 1.0)).unwrap();

        assert_relative_eq!(angles.costheta(), 0.0);
        assert_relative_eq!(angles.phi(), std::f64::consts::PI);
    }

    #[test]
    fn frame_axes_reject_degenerate_production_plane() {
        let err = FrameAxes::from_production_frame(
            Frame::Helicity,
            Vec4::new(0.0, 0.0, 5.0, 5.0),
            Vec4::new(0.0, 0.0, 1.0, 2.0),
            Vec4::new(0.0, 0.0, -1.0, 2.0),
            Vec3::zero(),
        );

        assert!(err.is_err());
    }
}
