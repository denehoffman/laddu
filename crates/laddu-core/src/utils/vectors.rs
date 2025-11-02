use auto_ops::{impl_op_ex, impl_op_ex_commutative};
use polars::prelude::*;

#[derive(Clone)]
pub struct Vec3([Expr; 3]);
impl From<[Expr; 3]> for Vec3 {
    fn from(value: [Expr; 3]) -> Self {
        Self([
            value[0].clone().cast(DataType::Float64),
            value[1].clone().cast(DataType::Float64),
            value[2].clone().cast(DataType::Float64),
        ])
    }
}
impl Vec3 {
    pub fn new<S: Into<PlSmallStr>>(name: S) -> Self {
        let name: PlSmallStr = name.into();
        Self([
            col(format!("{}_x", name)).cast(DataType::Float64),
            col(format!("{}_y", name)).cast(DataType::Float64),
            col(format!("{}_z", name)).cast(DataType::Float64),
        ])
    }
    pub fn alias<S: AsRef<str>>(&self, name: S) -> [Expr; 3] {
        let b = name.as_ref();
        [
            self.0[0].clone().alias(format!("{b}_x")),
            self.0[1].clone().alias(format!("{b}_y")),
            self.0[2].clone().alias(format!("{b}_z")),
        ]
    }
    pub fn x(&self) -> Expr {
        self.0[0].clone()
    }
    pub fn y(&self) -> Expr {
        self.0[1].clone()
    }
    pub fn z(&self) -> Expr {
        self.0[2].clone()
    }

    pub fn with_mass(&self, mass: &Expr) -> Vec4 {
        let e = (mass.clone().pow(2) + self.mag2()).sqrt();
        Vec4([self.x(), self.y(), self.z(), e])
    }

    pub fn with_energy(&self, energy: &Expr) -> Vec4 {
        Vec4([self.x(), self.y(), self.z(), energy.clone()])
    }

    pub fn dot(&self, other: &Self) -> Expr {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
    pub fn cross(&self, other: &Self) -> Self {
        Self([
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
        ])
    }
    pub fn mag2(&self) -> Expr {
        self.dot(self)
    }
    pub fn mag(&self) -> Expr {
        self.mag2().sqrt()
    }
    pub fn costheta(&self) -> Expr {
        self.z() / self.mag()
    }
    pub fn theta(&self) -> Expr {
        self.costheta().arccos()
    }
    pub fn phi(&self) -> Expr {
        self.y().arctan2(self.x())
    }
    pub fn unit(&self) -> Self {
        Self([
            self.x() / self.mag(),
            self.y() / self.mag(),
            self.z() / self.mag(),
        ])
    }
    pub fn add(&self, other: &Self) -> Self {
        Self([
            self.x() + other.x(),
            self.y() + other.y(),
            self.z() + other.z(),
        ])
    }
    pub fn scalar_add(&self, other: &Expr) -> Self {
        Self([
            self.x() + other.clone(),
            self.y() + other.clone(),
            self.z() + other.clone(),
        ])
    }
    pub fn sub(&self, other: &Self) -> Self {
        Self([
            self.x() - other.x(),
            self.y() - other.y(),
            self.z() - other.z(),
        ])
    }
    pub fn scalar_sub(&self, other: &Expr) -> Self {
        Self([
            self.x() - other.clone(),
            self.y() - other.clone(),
            self.z() - other.clone(),
        ])
    }
    pub fn scalar_rsub(&self, other: &Expr) -> Self {
        Self([
            other.clone() - self.x(),
            other.clone() - self.y(),
            other.clone() - self.z(),
        ])
    }
    pub fn mul(&self, other: &Expr) -> Self {
        Self([
            self.x() * other.clone(),
            self.y() * other.clone(),
            self.z() * other.clone(),
        ])
    }
    pub fn div(&self, other: &Expr) -> Self {
        Self([
            self.x() / other.clone(),
            self.y() / other.clone(),
            self.z() / other.clone(),
        ])
    }
    pub fn rdiv(&self, other: &Expr) -> Self {
        Self([
            other.clone() / self.x(),
            other.clone() / self.y(),
            other.clone() / self.z(),
        ])
    }
    pub fn neg(&self) -> Self {
        Self([-self.x(), -self.y(), -self.z()])
    }
}

impl_op_ex!(+ |a: &Vec3, b: &Vec3| -> Vec3 { a.add(b) });
impl_op_ex!(-|a: &Vec3, b: &Vec3| -> Vec3 { a.sub(b) });
impl_op_ex!(-|a: &Vec3| -> Vec3 { a.neg() });
impl_op_ex_commutative!(+ |a: &Vec3, b: &Expr| -> Vec3 { a.scalar_add(b) });
impl_op_ex!(-|a: &Vec3, b: &Expr| -> Vec3 { a.scalar_sub(b) });
impl_op_ex!(-|a: &Expr, b: &Vec3| -> Vec3 { b.scalar_rsub(a) });
impl_op_ex_commutative!(*|a: &Vec3, b: &Expr| -> Vec3 { a.mul(b) });
impl_op_ex!(/ |a: &Vec3, b: &Expr| -> Vec3 { a.div(b) });
impl_op_ex!(/ |a: &Expr, b: &Vec3| -> Vec3 { b.rdiv(a) });

#[derive(Clone)]
pub struct Vec4([Expr; 4]);
impl From<[Expr; 4]> for Vec4 {
    fn from(value: [Expr; 4]) -> Self {
        Self([
            value[0].clone().cast(DataType::Float64),
            value[1].clone().cast(DataType::Float64),
            value[2].clone().cast(DataType::Float64),
            value[3].clone().cast(DataType::Float64),
        ])
    }
}
impl Vec4 {
    pub fn new<S: Into<PlSmallStr>>(name: S) -> Self {
        let name: PlSmallStr = name.into();
        Self([
            col(format!("{}_px", name)).cast(DataType::Float64),
            col(format!("{}_py", name)).cast(DataType::Float64),
            col(format!("{}_pz", name)).cast(DataType::Float64),
            col(format!("{}_e", name)).cast(DataType::Float64),
        ])
    }

    pub fn sum<I, S>(constituents: I) -> Vec4
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let mut it = constituents.into_iter();
        let mut total = if let Some(first) = it.next() {
            Vec4::new(first)
        } else {
            Vec4([lit(0.0), lit(0.0), lit(0.0), lit(0.0)])
        };
        for n in it {
            total = total.add(&Vec4::new(n));
        }
        total
    }

    pub fn alias<S: AsRef<str>>(&self, name: S) -> [Expr; 4] {
        let b = name.as_ref();
        [
            self.0[0].clone().alias(format!("{b}_px")),
            self.0[1].clone().alias(format!("{b}_py")),
            self.0[2].clone().alias(format!("{b}_pz")),
            self.0[3].clone().alias(format!("{b}_e")),
        ]
    }
    pub fn px(&self) -> Expr {
        self.0[0].clone()
    }
    pub fn py(&self) -> Expr {
        self.0[1].clone()
    }
    pub fn pz(&self) -> Expr {
        self.0[2].clone()
    }
    pub fn e(&self) -> Expr {
        self.0[3].clone()
    }
    // let's get rid of "momentum"
    pub fn vec3(&self) -> Vec3 {
        Vec3([self.px(), self.py(), self.pz()])
    }
    pub fn beta(&self) -> Vec3 {
        self.vec3().div(&self.e())
    }
    pub fn gamma(&self) -> Expr {
        let e = self.e();
        let e2 = e.clone() * e.clone();
        let p2 = self.vec3().mag2();
        e / (e2 - p2).sqrt()
    }
    // let's also get rid of m and m2, unix philosophy
    pub fn mag2(&self) -> Expr {
        self.e() * self.e() - self.vec3().mag2()
    }
    pub fn mag(&self) -> Expr {
        self.mag2().sqrt()
    }
    pub fn boost(&self, beta: &Vec3) -> Self {
        let b2 = beta.dot(beta);
        let gamma = lit(1.0) / (lit(1.0) - b2.clone()).sqrt();
        let p3 = self.vec3()
            + (beta
                * (&((gamma.clone() - lit(1.0)) * self.vec3().dot(beta) / b2
                    + gamma.clone() * self.e())));
        Self([
            p3.x(),
            p3.y(),
            p3.z(),
            gamma * (self.e() + beta.dot(&self.vec3())),
        ])
    }
    pub fn add(&self, other: &Self) -> Self {
        Self([
            self.px() + other.px(),
            self.py() + other.py(),
            self.pz() + other.pz(),
            self.e() + other.e(),
        ])
    }
    pub fn sub(&self, other: &Self) -> Self {
        Self([
            self.px() - other.px(),
            self.py() - other.py(),
            self.pz() - other.pz(),
            self.e() - other.e(),
        ])
    }
    pub fn neg(&self) -> Self {
        Self([-self.px(), -self.py(), -self.pz(), -self.e()])
    }
}

impl_op_ex!(+ |a: &Vec4, b: &Vec4| -> Vec4 { a.add(b) });
impl_op_ex!(-|a: &Vec4, b: &Vec4| -> Vec4 { a.sub(b) });
impl_op_ex!(-|a: &Vec4| -> Vec4 { a.neg() });

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::tests::val1;

    /// Add {name}_x,{name}_y,{name}_z (Float64) with literal values to a DataFrame.
    pub fn add_vec3(df: &mut DataFrame, name: &str, v: [f64; 3]) -> Vec3 {
        let [x, y, z] = v;
        df.with_column(Series::new(format!("{}_x", name).into(), &[x]))
            .unwrap();
        df.with_column(Series::new(format!("{}_y", name).into(), &[y]))
            .unwrap();
        df.with_column(Series::new(format!("{}_z", name).into(), &[z]))
            .unwrap();
        Vec3::new(name)
    }

    /// Add {name}_x,{name}_y,{name}_z,{name}_e (Float64) with literal values to a DataFrame.
    pub fn add_vec4(df: &mut DataFrame, name: &str, v: [f64; 4]) -> Vec4 {
        let [px, py, pz, e] = v;
        df.with_column(Series::new(format!("{}_px", name).into(), &[px]))
            .unwrap();
        df.with_column(Series::new(format!("{}_py", name).into(), &[py]))
            .unwrap();
        df.with_column(Series::new(format!("{}_pz", name).into(), &[pz]))
            .unwrap();
        df.with_column(Series::new(format!("{}_e", name).into(), &[e]))
            .unwrap();
        Vec4::new(name)
    }

    #[test]
    fn test_vec_sums() {
        let mut df = DataFrame::empty();
        let a = add_vec3(&mut df, "a", [1.0, 2.0, 3.0]);
        let b = add_vec3(&mut df, "b", [4.0, 5.0, 6.0]);
        let lf = df.lazy();
        let res = lf.with_columns((a + b).alias("result")).collect().unwrap();
        assert_eq!(val1(&res, "result_x"), 5.0);
        assert_eq!(val1(&res, "result_y"), 7.0);
        assert_eq!(val1(&res, "result_z"), 9.0);
    }
    #[test]
    fn test_three_to_four_momentum_conversion() {
        let mut df = DataFrame::empty();
        let p3 = add_vec3(&mut df, "p3", [1.0, 2.0, 3.0]);
        let target_p4 = add_vec4(&mut df, "target_p4", [1.0, 2.0, 3.0, 10.0]);

        let p4_from_mass = p3.with_mass(&target_p4.mag());
        let p4_from_energy = p3.with_energy(&target_p4.e());
        let lf = df.lazy();
        let res = lf
            .with_columns(p4_from_mass.alias("p4_from_mass"))
            .with_columns(p4_from_energy.alias("p4_from_energy"))
            .collect()
            .unwrap();
        assert_eq!(val1(&res, "target_p4_e"), val1(&res, "p4_from_mass_e"));
        assert_eq!(val1(&res, "target_p4_px"), val1(&res, "p4_from_mass_px"));
        assert_eq!(val1(&res, "target_p4_py"), val1(&res, "p4_from_mass_py"));
        assert_eq!(val1(&res, "target_p4_pz"), val1(&res, "p4_from_mass_pz"));
        assert_eq!(val1(&res, "target_p4_e"), val1(&res, "p4_from_energy_e"));
        assert_eq!(val1(&res, "target_p4_px"), val1(&res, "p4_from_energy_px"));
        assert_eq!(val1(&res, "target_p4_py"), val1(&res, "p4_from_energy_py"));
        assert_eq!(val1(&res, "target_p4_pz"), val1(&res, "p4_from_energy_pz"));
    }
    #[test]
    fn test_four_momentum_basics() {
        let mut df = DataFrame::empty();
        let p = add_vec4(&mut df, "p", [3.0, 4.0, 5.0, 10.0]);
        let lf = df.lazy();
        let res = lf
            .with_columns(p.vec3().alias("p"))
            .with_columns(p.beta().alias("beta"))
            .with_columns([
                p.mag().alias("m"),
                p.mag2().alias("m2"),
                p.gamma().alias("gamma"),
            ])
            .collect()
            .unwrap();
        assert_eq!(val1(&res, "p_e"), 10.0);
        assert_eq!(val1(&res, "p_px"), 3.0);
        assert_eq!(val1(&res, "p_py"), 4.0);
        assert_eq!(val1(&res, "p_pz"), 5.0);
        assert_eq!(val1(&res, "p_x"), 3.0);
        assert_eq!(val1(&res, "p_y"), 4.0);
        assert_eq!(val1(&res, "p_z"), 5.0);
        assert_relative_eq!(val1(&res, "beta_x"), 0.3);
        assert_relative_eq!(val1(&res, "beta_y"), 0.4);
        assert_relative_eq!(val1(&res, "beta_z"), 0.5);
        assert_relative_eq!(val1(&res, "m"), 50.0_f64.sqrt());
        assert_relative_eq!(val1(&res, "m2"), 50.0);
        assert_relative_eq!(val1(&res, "gamma"), 2.0_f64.sqrt());
    }
    #[test]
    fn test_three_momentum_basics() {
        let mut df = DataFrame::empty();
        let p = add_vec4(&mut df, "p", [3.0, 4.0, 5.0, 10.0]);
        let q = add_vec4(&mut df, "q", [1.2, -3.4, 7.6, 0.0]);
        let p3 = p.vec3();
        let q3 = q.vec3();
        let lf = df.lazy();
        let res = lf
            .with_columns([
                p3.mag().alias("m"),
                p3.mag2().alias("m2"),
                p3.costheta().alias("costheta"),
                p3.theta().alias("theta"),
                p3.phi().alias("phi"),
            ])
            .with_columns(p3.unit().alias("u3"))
            .with_columns(p3.cross(&q3).alias("cross"))
            .collect()
            .unwrap();
        assert_relative_eq!(val1(&res, "m"), 50.0_f64.sqrt());
        assert_relative_eq!(val1(&res, "m2"), 50.0);
        assert_relative_eq!(val1(&res, "costheta"), 5.0 / 50.0_f64.sqrt());
        assert_relative_eq!(val1(&res, "theta"), (5.0 / 50.0_f64.sqrt()).acos());
        assert_relative_eq!(val1(&res, "phi"), 4.0_f64.atan2(3.0));
        assert_relative_eq!(val1(&res, "u3_x"), 3.0 / 50.0_f64.sqrt());
        assert_relative_eq!(val1(&res, "u3_y"), 4.0 / 50.0_f64.sqrt());
        assert_relative_eq!(val1(&res, "u3_z"), 5.0 / 50.0_f64.sqrt());
        assert_relative_eq!(val1(&res, "cross_x"), 47.4);
        assert_relative_eq!(val1(&res, "cross_y"), -16.8);
        assert_relative_eq!(val1(&res, "cross_z"), -15.0);
    }

    #[test]
    fn test_boost_com() {
        let mut df = DataFrame::empty();
        let p = add_vec4(&mut df, "p", [3.0, 4.0, 5.0, 10.0]);
        let lf = df.lazy();
        let res = lf
            .with_columns(p.boost(&-p.beta()).vec3().alias("zero"))
            .collect()
            .unwrap();
        assert_relative_eq!(val1(&res, "zero_x"), 0.0);
        assert_relative_eq!(val1(&res, "zero_y"), 0.0);
        assert_relative_eq!(val1(&res, "zero_z"), 0.0);
    }

    #[test]
    fn test_boost() {
        let mut df = DataFrame::empty();
        let pa = add_vec4(&mut df, "pa", [3.0, 4.0, 5.0, 10.0]);
        let pb = add_vec4(&mut df, "pb", [3.4, 2.3, 1.2, 9.0]);
        let lf = df.lazy();
        let res = lf
            .with_columns(pa.boost(&-pb.beta()).alias("boosted"))
            .collect()
            .unwrap();
        assert_relative_eq!(val1(&res, "boosted_e"), 8.157632144622882);
        assert_relative_eq!(val1(&res, "boosted_px"), -0.6489200627053444);
        assert_relative_eq!(val1(&res, "boosted_py"), 1.5316128987581492);
        assert_relative_eq!(val1(&res, "boosted_pz"), 3.712145860221643);
    }
}
