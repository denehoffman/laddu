use nalgebra::{Vector3, Vector4, VectorView, U1, U3, U4};

use crate::Float;

/// An object which behaves as a four-vector.
pub trait FourVector {
    /// The magnitude of the vector (with $`+---`$ signature).
    fn mag(&self) -> Float;
    /// The squared magnitude of the vector (with $`+---`$ signature).
    fn mag2(&self) -> Float;
    /// Yields the three-vector part.
    fn vec3(&self) -> VectorView<Float, U3, U1, U4>;
    /// Gives the vector boosted along a $`\vec{\beta}`$ vector.
    fn boost(&self, beta: &Vector3<Float>) -> Self;
}

/// An object which behaves as a four-momentum.
pub trait FourMomentum: FourVector {
    /// Energy
    fn e(&self) -> Float;
    /// Momentum in the $`x`$-direction
    fn px(&self) -> Float;
    /// Momentum in the $`y`$-direction
    fn py(&self) -> Float;
    /// Momentum in the $`z`$-direction
    fn pz(&self) -> Float;
    /// The three-momentum
    fn momentum(&self) -> VectorView<Float, U3, U1, U4>;
    /// The $`\vec{\beta}`$ vector $`\frac{\vec{p}}{E}`$.
    fn beta(&self) -> Vector3<Float>;
    /// The mass of the corresponding object.
    fn m(&self) -> Float;
    /// The squared mass of the corresponding object.
    fn m2(&self) -> Float;
    /// Yields the vector boosted along the direction of another [`FourMomentum`].
    fn boost_along(&self, other: &Self) -> Self;
    /// Creates a [`FourMomentum`] from a [`ThreeMomentum`] and a mass using $`E^2 = m^2 + p^2`$.
    fn from_momentum(momentum: &dyn ThreeMomentum, mass: Float) -> Self;
    /// Pretty-prints the four-momentum.
    fn to_p4_string(&self) -> String {
        format!(
            "[e = {:.5}; p = ({:.5}, {:.5}, {:.5}); m = {:.5}]",
            self.e(),
            self.px(),
            self.py(),
            self.pz(),
            self.m()
        )
    }
}

/// Useful methods for three-dimensional vectors.
pub trait ThreeVector {
    /// The magnitude of the vector.
    fn mag(&self) -> Float;
    /// The squared magnitude of the vector.
    fn mag2(&self) -> Float;
    /// The cosine of the polar angle $`\theta`$.
    fn costheta(&self) -> Float;
    /// The polar angle $`\theta`$.
    fn theta(&self) -> Float;
    /// The azimuthal angle $`\phi`$.
    fn phi(&self) -> Float;
    /// Creates a unit vector in the direction of the input vector.
    fn unit(&self) -> Vector3<Float>;
}

/// Additional methods for treating a [`ThreeVector`] as a three-momentum.
pub trait ThreeMomentum: ThreeVector {
    /// Momentum in the $`x`$-direction
    fn px(&self) -> Float;
    /// Momentum in the $`y`$-direction
    fn py(&self) -> Float;
    /// Momentum in the $`z`$-direction
    fn pz(&self) -> Float;
}

impl FourVector for Vector4<Float> {
    fn mag(&self) -> Float {
        Float::sqrt(self.mag2())
    }

    fn mag2(&self) -> Float {
        self[0] * self[0] - (self[1] * self[1] + self[2] * self[2] + self[3] * self[3])
    }

    fn boost(&self, beta: &Vector3<Float>) -> Self {
        let b2 = beta.dot(beta);
        let gamma = 1.0 / Float::sqrt(1.0 - b2);
        let p3 =
            self.momentum() + beta * ((gamma - 1.0) * self.vec3().dot(beta) / b2 - gamma * self[0]);
        Vector4::new(gamma * self[0] - beta.dot(&self.vec3()), p3.x, p3.y, p3.z)
    }

    fn vec3(&self) -> VectorView<Float, U3, U1, U4> {
        self.fixed_rows::<3>(1)
    }
}

impl FourMomentum for Vector4<Float> {
    fn e(&self) -> Float {
        self[0]
    }

    fn px(&self) -> Float {
        self[1]
    }

    fn py(&self) -> Float {
        self[2]
    }

    fn pz(&self) -> Float {
        self[3]
    }

    fn momentum(&self) -> VectorView<Float, U3, U1, U4> {
        self.vec3()
    }

    fn beta(&self) -> Vector3<Float> {
        self.momentum().unscale(self.e())
    }

    fn m(&self) -> Float {
        self.mag()
    }

    fn m2(&self) -> Float {
        self.mag2()
    }

    fn boost_along(&self, other: &Self) -> Self {
        self.boost(&other.beta())
    }

    fn from_momentum(momentum: &dyn ThreeMomentum, mass: Float) -> Self {
        let e = Float::sqrt(mass.powi(2) + momentum.mag2());
        Self::new(e, momentum.px(), momentum.py(), momentum.pz())
    }
}

impl ThreeMomentum for Vector3<Float> {
    fn px(&self) -> Float {
        self[0]
    }

    fn py(&self) -> Float {
        self[1]
    }

    fn pz(&self) -> Float {
        self[2]
    }
}

impl ThreeVector for Vector3<Float> {
    fn mag(&self) -> Float {
        Float::sqrt(self.mag2())
    }

    fn mag2(&self) -> Float {
        self.dot(self)
    }

    fn costheta(&self) -> Float {
        self.z / self.mag()
    }

    fn theta(&self) -> Float {
        Float::acos(self.costheta())
    }

    fn phi(&self) -> Float {
        Float::atan2(self.y, self.x)
    }

    fn unit(&self) -> Vector3<Float> {
        self.unscale(self.mag())
    }
}

impl<'a> ThreeMomentum for VectorView<'a, Float, U3, U1, U4> {
    fn px(&self) -> Float {
        self[0]
    }

    fn py(&self) -> Float {
        self[1]
    }

    fn pz(&self) -> Float {
        self[2]
    }
}

impl<'a> ThreeVector for VectorView<'a, Float, U3, U1, U4> {
    fn mag(&self) -> Float {
        Float::sqrt(self.mag2())
    }

    fn mag2(&self) -> Float {
        self.dot(self)
    }

    fn costheta(&self) -> Float {
        self.z / self.mag()
    }

    fn theta(&self) -> Float {
        Float::acos(self.costheta())
    }

    fn phi(&self) -> Float {
        Float::atan2(self.y, self.x)
    }

    fn unit(&self) -> Vector3<Float> {
        self.to_owned().unscale(self.mag())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::vector;

    use super::*;

    #[test]
    fn test_four_momentum_basics() {
        let p = vector![10.0, 3.0, 4.0, 5.0];
        assert_eq!(p.e(), 10.0);
        assert_eq!(p.px(), 3.0);
        assert_eq!(p.py(), 4.0);
        assert_eq!(p.pz(), 5.0);
        assert_eq!(p.momentum().px(), 3.0);
        assert_eq!(p.momentum().py(), 4.0);
        assert_eq!(p.momentum().pz(), 5.0);
        assert_relative_eq!(p.beta().x, 0.3);
        assert_relative_eq!(p.beta().y, 0.4);
        assert_relative_eq!(p.beta().z, 0.5);
        assert_relative_eq!(p.m2(), 50.0);
        assert_relative_eq!(p.m(), Float::sqrt(50.0));
        assert_eq!(
            format!("{}", p.to_p4_string()),
            "[e = 10.00000; p = (3.00000, 4.00000, 5.00000); m = 7.07107]"
        );
    }

    #[test]
    fn test_three_momentum_basics() {
        let p = vector![10.0, 3.0, 4.0, 5.0];
        let q = vector![0.0, 1.2, -3.4, 7.6];
        let p3_view = p.momentum();
        let q3_view = q.momentum();
        assert_eq!(p3_view.px(), 3.0);
        assert_eq!(p3_view.py(), 4.0);
        assert_eq!(p3_view.pz(), 5.0);
        assert_relative_eq!(p3_view.mag2(), 50.0);
        assert_relative_eq!(p3_view.mag(), Float::sqrt(50.0));
        assert_relative_eq!(p3_view.costheta(), 5.0 / Float::sqrt(50.0));
        assert_relative_eq!(p3_view.theta(), Float::acos(5.0 / Float::sqrt(50.0)));
        assert_relative_eq!(p3_view.phi(), Float::atan2(4.0, 3.0));
        assert_relative_eq!(
            p3_view.unit(),
            vector![3.0, 4.0, 5.0].unscale(Float::sqrt(50.0))
        );
        assert_relative_eq!(p3_view.cross(&q3_view), vector![47.4, -16.8, -15.0]);
        let p3 = vector![3.0, 4.0, 5.0];
        let q3 = vector![1.2, -3.4, 7.6];
        assert_eq!(p3.px(), 3.0);
        assert_eq!(p3.py(), 4.0);
        assert_eq!(p3.pz(), 5.0);
        assert_relative_eq!(p3.mag2(), 50.0);
        assert_relative_eq!(p3.mag(), Float::sqrt(50.0));
        assert_relative_eq!(p3.costheta(), 5.0 / Float::sqrt(50.0));
        assert_relative_eq!(p3.theta(), Float::acos(5.0 / Float::sqrt(50.0)));
        assert_relative_eq!(p3.phi(), Float::atan2(4.0, 3.0));
        assert_relative_eq!(p3.unit(), vector![3.0, 4.0, 5.0].unscale(Float::sqrt(50.0)));
        assert_relative_eq!(p3.cross(&q3), vector![47.4, -16.8, -15.0]);
    }

    #[test]
    fn test_boost_com() {
        let p = vector![10.0, 3.0, 4.0, 5.0];
        let zero = p.boost_along(&(-p));
        assert_relative_eq!(zero[1], 0.0);
        assert_relative_eq!(zero[2], 0.0);
        assert_relative_eq!(zero[3], 0.0);
    }
}
