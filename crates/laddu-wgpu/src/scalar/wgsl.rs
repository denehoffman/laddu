//! WGPU-specific shader vocabulary and lowering entry points.
//!
//! Kernel traversal and shape facts remain owned by `laddu-kernel`; this
//! module owns only the target spelling of scalar, complex, and aggregate
//! values.  Keeping that vocabulary here means another backend never has to
//! depend on WGSL strings.

use laddu_expr::{BinaryOp, UnaryOp};
use laddu_kernel::ir::{
    CacheKernelIr, GradientKernelIr, KernelInstruction, KernelValue, KernelValueId,
    KernelValueKind, ScalarKernelIr,
};

use super::bindings::Binding;
use super::memory::WORKGROUP_SIZE;
use crate::{WgpuError, WgpuPrecision, WgpuResult};
use laddu_compile::CacheLayout;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScalarType {
    F32,
    F64,
}

impl ScalarType {
    pub(crate) const fn wgsl(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WgslDialect {
    scalar: ScalarType,
}

impl WgslDialect {
    pub(crate) fn for_precision(precision: WgpuPrecision) -> WgpuResult<Self> {
        let scalar = match precision {
            WgpuPrecision::F32 => ScalarType::F32,
            WgpuPrecision::F64 => ScalarType::F64,
            WgpuPrecision::Auto => return Err(WgpuError::UnsupportedKernelPrecision(precision)),
        };
        Ok(Self { scalar })
    }

    pub(crate) const fn scalar_type(self) -> &'static str {
        self.scalar.wgsl()
    }

    pub(crate) const fn complex_type(self) -> &'static str {
        match self.scalar {
            ScalarType::F32 => "vec2<f32>",
            ScalarType::F64 => "vec2<f64>",
        }
    }
}

impl crate::scalar::WgpuScalarKernel {
    fn scalar_prelude(precision: crate::WgpuPrecision) -> &'static str {
        match precision {
            crate::WgpuPrecision::F32 => {
                "fn scalar_sin(x: f32) -> f32 { return sin(x); }\n\
fn scalar_cos(x: f32) -> f32 { return cos(x); }\n\
fn scalar_exp(x: f32) -> f32 { return exp(x); }\n\
fn scalar_sinh(x: f32) -> f32 { return sinh(x); }\n\
fn scalar_cosh(x: f32) -> f32 { return cosh(x); }\n\
fn scalar_atan2(y: f32, x: f32) -> f32 { return atan2(y, x); }\n\
fn scalar_log(x: f32) -> f32 { return log(x); }\n\
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }\n\
fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { let d=b.x*b.x+b.y*b.y; return vec2((a.x*b.x+a.y*b.y)/d, (a.y*b.x-a.x*b.y)/d); }\n\
fn cnorm(z: vec2<f32>) -> f32 { return z.x*z.x+z.y*z.y; }\n\
fn cabs(z: vec2<f32>) -> f32 { let x=abs(z.x); let y=abs(z.y); let hi=max(x,y); let lo=min(x,y); let ratio=lo/hi; return select(hi*sqrt(1.0+ratio*ratio), 0.0, hi == 0.0); }\n\
fn csqrt(z: vec2<f32>) -> vec2<f32> { let m=cabs(z); let re=sqrt(max(0.0, 0.5*(m+z.x))); let im=sqrt(max(0.0, 0.5*(m-z.x))); return vec2(re, select(-im, im, z.y >= 0.0)); }\n\
fn cexp(z: vec2<f32>) -> vec2<f32> { let e=scalar_exp(z.x); return vec2(e*scalar_cos(z.y), e*scalar_sin(z.y)); }\n\
fn csin(z: vec2<f32>) -> vec2<f32> { return vec2(scalar_sin(z.x)*scalar_cosh(z.y), scalar_cos(z.x)*scalar_sinh(z.y)); }\n\
fn ccos(z: vec2<f32>) -> vec2<f32> { return vec2(scalar_cos(z.x)*scalar_cosh(z.y), -scalar_sin(z.x)*scalar_sinh(z.y)); }\n\
fn clog(z: vec2<f32>) -> vec2<f32> { return vec2(scalar_log(cabs(z)), scalar_atan2(z.y, z.x)); }\n\
fn cpowi(z: vec2<f32>, exponent: i32) -> vec2<f32> { var result=vec2<f32>(1.0, 0.0); var base=z; var n=abs(exponent); loop { if (n == 0) { break; } if ((n & 1) == 1) { result=cmul(result, base); } base=cmul(base, base); n=n/2; } if (exponent < 0) { return cdiv(vec2<f32>(1.0, 0.0), result); } return result; }\n"
            }
            crate::WgpuPrecision::F64 => {
                r#"fn scalar_sincos(x: f64) -> vec2<f64> {
    let ax = abs(x);
    let k = floor(ax * 0.63661977236758134308 + 0.5);
    let r = (ax - k * 1.57079632673412561417) - k * 6.07710050650619224932e-11;
    let z = r * r;
    let s = r + r * z * (-1.66666666666666324348e-1 + z * (8.33333333332248946124e-3 + z * (-1.98412698298579493134e-4 + z * (2.75573137070700676789e-6 + z * (-2.50507602534068634195e-8 + z * 1.58969099521155010221e-10)))));
    let c = 1.0 - 0.5 * z + z * z * (4.16666666666666019037e-2 + z * (-1.38888888888741095749e-3 + z * (2.48015872894767294178e-5 + z * (-2.75573143513906633035e-7 + z * (2.08757232129817482790e-9 + z * -1.13596475577881948265e-11)))));
    let quadrant = i32(k - 4.0 * floor(k * 0.25));
    var result = vec2<f64>(s, c);
    if (quadrant == 1) { result = vec2<f64>(c, -s); }
    if (quadrant == 2) { result = vec2<f64>(-s, -c); }
    if (quadrant == 3) { result = vec2<f64>(-c, s); }
    if (x < 0.0) { result.x = -result.x; }
    return result;
}
fn scalar_sin(x: f64) -> f64 { return scalar_sincos(x).x; }
fn scalar_cos(x: f64) -> f64 { return scalar_sincos(x).y; }
fn scalar_exp(x: f64) -> f64 {
    if (x != x) { return x; }
    let maximum: f64 = 1.79769313486231570815e308;
    if (x > 7.09782712893383973096e2) { return maximum * maximum; }
    if (x < -7.45133219101941108420e2) { return 0.0; }
    let k = floor(x * 1.44269504088896340736 + 0.5);
    let r = (x - k * 6.93147180369123816490e-1) - k * 1.90821492927058770002e-10;
    let z = r * r;
    let p = 1.66666666666666019037e-1 + z * (-2.77777777770155933842e-3 + z * (6.61375632143793436117e-5 + z * (-1.65339022054652515390e-6 + z * 4.13813679705723846039e-8)));
    let c = r - z * p;
    let reduced = 1.0 + r + r * c / (2.0 - c);
    var n = i32(k);
    var factor: f64 = 1.0;
    loop {
        if (n == 0) { break; }
        if (n > 0) {
            factor *= 2.0;
            n -= 1;
        } else {
            factor *= 0.5;
            n += 1;
        }
    }
    return reduced * factor;
}
fn scalar_sinh(x: f64) -> f64 {
    let ax = abs(x);
    if (ax <= 1.0) {
        let z = x * x;
        return x * (1.0 + z * (1.0 / 6.0 + z * (1.0 / 120.0 + z * (1.0 / 5040.0 + z * (1.0 / 362880.0 + z * (1.0 / 39916800.0 + z * (1.0 / 6227020800.0 + z * (1.0 / 1307674368000.0 + z / 355687428096000.0))))))));
    }
    let e = scalar_exp(ax);
    let value = 0.5 * (e - 1.0 / e);
    return select(-value, value, x >= 0.0);
}
fn scalar_cosh(x: f64) -> f64 {
    let ax = abs(x);
    if (ax <= 1.0) {
        let z = x * x;
        return 1.0 + z * (0.5 + z * (1.0 / 24.0 + z * (1.0 / 720.0 + z * (1.0 / 40320.0 + z * (1.0 / 3628800.0 + z * (1.0 / 479001600.0 + z * (1.0 / 87178291200.0 + z / 20922789888000.0)))))));
    }
    let e = scalar_exp(ax);
    return 0.5 * (e + 1.0 / e);
}
fn scalar_atan(x: f64) -> f64 {
    let ax = abs(x);
    var r = ax;
    var offset: f64 = 0.0;
    var subtract = false;
    if (ax > 2.41421356237309504880) {
        r = 1.0 / ax;
        offset = 1.57079632679489661923;
        subtract = true;
    } else if (ax > 0.41421356237309504880) {
        r = (ax - 1.0) / (ax + 1.0);
        offset = 0.78539816339744830962;
    }
    let z = r * r;
    var p: f64 = -1.0 / 39.0;
    p = 1.0 / 37.0 + z * p;
    p = -1.0 / 35.0 + z * p;
    p = 1.0 / 33.0 + z * p;
    p = -1.0 / 31.0 + z * p;
    p = 1.0 / 29.0 + z * p;
    p = -1.0 / 27.0 + z * p;
    p = 1.0 / 25.0 + z * p;
    p = -1.0 / 23.0 + z * p;
    p = 1.0 / 21.0 + z * p;
    p = -1.0 / 19.0 + z * p;
    p = 1.0 / 17.0 + z * p;
    p = -1.0 / 15.0 + z * p;
    p = 1.0 / 13.0 + z * p;
    p = -1.0 / 11.0 + z * p;
    p = 1.0 / 9.0 + z * p;
    p = -1.0 / 7.0 + z * p;
    p = 1.0 / 5.0 + z * p;
    p = -1.0 / 3.0 + z * p;
    let a = r + r * z * p;
    let value = select(offset + a, offset - a, subtract);
    return select(-value, value, x >= 0.0);
}
fn scalar_atan2(y: f64, x: f64) -> f64 {
    if (x > 0.0) { return scalar_atan(y / x); }
    if (x < 0.0) {
        let a = scalar_atan(y / x);
        return select(a - 3.14159265358979323846, a + 3.14159265358979323846, y >= 0.0);
    }
    if (y > 0.0) { return 1.57079632679489661923; }
    if (y < 0.0) { return -1.57079632679489661923; }
    return 0.0;
}
fn scalar_log(x: f64) -> f64 {
    if (x != x) { return x; }
    let maximum: f64 = 1.79769313486231570815e308;
    if (x > maximum) { return x; }
    let zero = x - x;
    if (x == 0.0) { return -1.0 / zero; }
    if (x < 0.0) { return zero / zero; }
    var m = x;
    var e: f64 = 0.0;
    if (m >= 1.34078079299425970996e154) { m *= 7.45834073120020674329e-155; e += 512.0; }
    if (m >= 1.15792089237316195424e77) { m *= 8.63616855509444462539e-78; e += 256.0; }
    if (m >= 3.40282366920938463463e38) { m *= 2.93873587705571876992e-39; e += 128.0; }
    if (m >= 1.84467440737095516160e19) { m *= 5.42101086242752217004e-20; e += 64.0; }
    if (m >= 4294967296.0) { m *= 2.32830643653869628906e-10; e += 32.0; }
    if (m >= 65536.0) { m *= 1.52587890625e-5; e += 16.0; }
    if (m >= 256.0) { m *= 0.00390625; e += 8.0; }
    if (m >= 16.0) { m *= 0.0625; e += 4.0; }
    if (m >= 4.0) { m *= 0.25; e += 2.0; }
    if (m >= 2.0) { m *= 0.5; e += 1.0; }
    if (m < 7.45834073120020674329e-155) { m *= 1.34078079299425970996e154; e -= 512.0; }
    if (m < 8.63616855509444462539e-78) { m *= 1.15792089237316195424e77; e -= 256.0; }
    if (m < 2.93873587705571876992e-39) { m *= 3.40282366920938463463e38; e -= 128.0; }
    if (m < 5.42101086242752217004e-20) { m *= 1.84467440737095516160e19; e -= 64.0; }
    if (m < 2.32830643653869628906e-10) { m *= 4294967296.0; e -= 32.0; }
    if (m < 1.52587890625e-5) { m *= 65536.0; e -= 16.0; }
    if (m < 0.00390625) { m *= 256.0; e -= 8.0; }
    if (m < 0.0625) { m *= 16.0; e -= 4.0; }
    if (m < 0.25) { m *= 4.0; e -= 2.0; }
    if (m < 0.5) { m *= 2.0; e -= 1.0; }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    var p: f64 = 1.0 / 31.0;
    p = 1.0 / 29.0 + z2 * p;
    p = 1.0 / 27.0 + z2 * p;
    p = 1.0 / 25.0 + z2 * p;
    p = 1.0 / 23.0 + z2 * p;
    p = 1.0 / 21.0 + z2 * p;
    p = 1.0 / 19.0 + z2 * p;
    p = 1.0 / 17.0 + z2 * p;
    p = 1.0 / 15.0 + z2 * p;
    p = 1.0 / 13.0 + z2 * p;
    p = 1.0 / 11.0 + z2 * p;
    p = 1.0 / 9.0 + z2 * p;
    p = 1.0 / 7.0 + z2 * p;
    p = 1.0 / 5.0 + z2 * p;
    p = 1.0 / 3.0 + z2 * p;
    let log_m = 2.0 * z * (1.0 + z2 * p);
    return log_m + e * 0.69314718055994530942;
}
fn cmul(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> { return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
fn cdiv(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> { let d=b.x*b.x+b.y*b.y; return vec2((a.x*b.x+a.y*b.y)/d, (a.y*b.x-a.x*b.y)/d); }
fn cnorm(z: vec2<f64>) -> f64 { return z.x*z.x+z.y*z.y; }
fn cabs(z: vec2<f64>) -> f64 { let x=abs(z.x); let y=abs(z.y); let hi=max(x,y); let lo=min(x,y); let ratio=lo/hi; return select(hi*sqrt(1.0+ratio*ratio), 0.0, hi == 0.0); }
fn csqrt(z: vec2<f64>) -> vec2<f64> { let m=cabs(z); let re=sqrt(max(0.0, 0.5*(m+z.x))); let im=sqrt(max(0.0, 0.5*(m-z.x))); return vec2(re, select(-im, im, z.y >= 0.0)); }
fn cexp(z: vec2<f64>) -> vec2<f64> { let e=scalar_exp(z.x); return vec2(e*scalar_cos(z.y), e*scalar_sin(z.y)); }
fn csin(z: vec2<f64>) -> vec2<f64> { return vec2(scalar_sin(z.x)*scalar_cosh(z.y), scalar_cos(z.x)*scalar_sinh(z.y)); }
fn ccos(z: vec2<f64>) -> vec2<f64> { return vec2(scalar_cos(z.x)*scalar_cosh(z.y), -scalar_sin(z.x)*scalar_sinh(z.y)); }
fn clog(z: vec2<f64>) -> vec2<f64> { return vec2(scalar_log(cabs(z)), scalar_atan2(z.y, z.x)); }
fn cpowi(z: vec2<f64>, exponent: i32) -> vec2<f64> { var result=vec2<f64>(1.0, 0.0); var base=z; var n=abs(exponent); loop { if (n == 0) { break; } if ((n & 1) == 1) { result=cmul(result, base); } base=cmul(base, base); n=n/2; } if (exponent < 0) { return cdiv(vec2<f64>(1.0, 0.0), result); } return result; }
"#
            }
            crate::WgpuPrecision::Auto => unreachable!("kernel precision is resolved"),
        }
    }

    fn aggregate(
        elements: impl IntoIterator<Item = String>,
        width: usize,
        precision: crate::WgpuPrecision,
    ) -> String {
        format!(
            "array<{}, {width}>({})",
            WgslDialect::for_precision(precision)
                .expect("kernel precision is resolved")
                .complex_type(),
            elements.into_iter().collect::<Vec<_>>().join(", ")
        )
    }

    fn emit_unary(op: UnaryOp, input: KernelValueId, complex: &str) -> String {
        let input = format!("v{}", input.index());
        match op {
            UnaryOp::Neg => format!("-{input}"),
            UnaryOp::Real => format!("{complex}({input}.x, 0.0)"),
            UnaryOp::Imag => format!("{complex}({input}.y, 0.0)"),
            UnaryOp::Conj => format!("{complex}({input}.x, -{input}.y)"),
            UnaryOp::NormSqr => format!("{complex}(cnorm({input}), 0.0)"),
            UnaryOp::Sqrt => format!("csqrt({input})"),
            UnaryOp::Exp => format!("cexp({input})"),
            UnaryOp::Sin => format!("csin({input})"),
            UnaryOp::Cos => format!("ccos({input})"),
            UnaryOp::Log => format!("clog({input})"),
            UnaryOp::PowI(power) => format!("cpowi({input}, {power})"),
        }
    }

    fn emit_binary(op: BinaryOp, lhs: KernelValueId, rhs: KernelValueId, complex: &str) -> String {
        let lhs = format!("v{}", lhs.index());
        let rhs = format!("v{}", rhs.index());
        match op {
            BinaryOp::Add => format!("{lhs} + {rhs}"),
            BinaryOp::Sub => format!("{lhs} - {rhs}"),
            BinaryOp::Mul => format!("cmul({lhs}, {rhs})"),
            BinaryOp::Div => format!("cdiv({lhs}, {rhs})"),
            BinaryOp::Atan2 => format!("{complex}(scalar_atan2({lhs}.x, {rhs}.x), 0.0)"),
        }
    }

    fn emit_variadic(ids: &[KernelValueId], multiply: bool) -> WgpuResult<String> {
        let mut values = ids.iter().map(|id| format!("v{}", id.index()));
        let Some(first) = values.next() else {
            return Err(WgpuError::UnsupportedInstruction(
                "empty variadic instruction".into(),
            ));
        };
        Ok(values.fold(first, |left, right| {
            if multiply {
                format!("cmul({left}, {right})")
            } else {
                format!("{left} + {right}")
            }
        }))
    }

    fn emit_dot(
        values: &[KernelValue],
        lhs: KernelValueId,
        rhs: KernelValueId,
    ) -> WgpuResult<String> {
        let KernelValueKind::Vector { len } = values[lhs.index()].kind else {
            unreachable!("dot-product IR was validated")
        };
        (0..len)
            .map(|element| {
                format!(
                    "cmul(v{}[{element}], v{}[{element}])",
                    lhs.index(),
                    rhs.index()
                )
            })
            .reduce(|left, right| format!("{left} + {right}"))
            .ok_or_else(|| WgpuError::UnsupportedInstruction("empty dot product".into()))
    }

    fn emit_matrix_product(
        values: &[KernelValue],
        lhs: KernelValueId,
        rhs: KernelValueId,
        precision: crate::WgpuPrecision,
    ) -> WgpuResult<String> {
        let KernelValueKind::Matrix { rows, cols: inner } = values[lhs.index()].kind else {
            unreachable!("matrix-matrix IR was validated")
        };
        let KernelValueKind::Matrix { cols, .. } = values[rhs.index()].kind else {
            unreachable!("matrix-matrix IR was validated")
        };
        let elements = (0..rows).flat_map(|row| {
            (0..cols).map(move |col| {
                (0..inner)
                    .map(|element| {
                        format!(
                            "cmul(v{}[{}], v{}[{}])",
                            lhs.index(),
                            row * inner + element,
                            rhs.index(),
                            element * cols + col
                        )
                    })
                    .reduce(|left, right| format!("{left} + {right}"))
                    .ok_or_else(|| WgpuError::UnsupportedInstruction("empty matrix product".into()))
            })
        });
        let elements = elements.collect::<WgpuResult<Vec<_>>>()?;
        let width = rows * cols;
        Ok(Self::aggregate(elements, width, precision))
    }

    fn emit_matrix_vector(
        values: &[KernelValue],
        matrix: KernelValueId,
        vector: KernelValueId,
        precision: crate::WgpuPrecision,
    ) -> WgpuResult<String> {
        let KernelValueKind::Matrix { rows, cols } = values[matrix.index()].kind else {
            unreachable!("matrix-vector IR was validated")
        };
        let rows = (0..rows).map(|row| {
            (0..cols)
                .map(|col| {
                    format!(
                        "cmul(v{}[{}], v{}[{col}])",
                        matrix.index(),
                        row * cols + col,
                        vector.index()
                    )
                })
                .reduce(|left, right| format!("{left} + {right}"))
                .ok_or_else(|| WgpuError::UnsupportedInstruction("empty matrix row".into()))
        });
        let rows = rows.collect::<WgpuResult<Vec<_>>>()?;
        let width = rows.len();
        Ok(Self::aggregate(rows, width, precision))
    }

    fn emit_values(
        values: &[KernelValue],
        precision: crate::WgpuPrecision,
        cached: impl Fn(usize, KernelValueKind) -> String,
    ) -> WgpuResult<String> {
        let mut source = String::new();
        let v = |id: KernelValueId| format!("v{}", id.index());
        let complex = WgslDialect::for_precision(precision)
            .expect("kernel precision is resolved")
            .complex_type();
        for (index, value) in values.iter().enumerate() {
            if let KernelInstruction::Solve { matrix, rhs } = &value.instruction {
                let KernelValueKind::Matrix { rows, cols } = values[matrix.index()].kind else {
                    unreachable!("solve IR was validated")
                };
                if rows > 16 {
                    return Err(WgpuError::SolveDimensionTooLarge { dimension: rows });
                }
                debug_assert_eq!(rows, cols);
                source.push_str(&Self::emit_solve(
                    index,
                    &v(*matrix),
                    &v(*rhs),
                    rows,
                    precision,
                ));
                continue;
            }
            let expr = match &value.instruction {
                KernelInstruction::Cached(slot) => cached(*slot, value.kind),
                KernelInstruction::RealConstant(x) => format!("{complex}({x:?}, 0.0)"),
                KernelInstruction::ComplexConstant(x) => {
                    format!("{complex}({:?}, {:?})", x.re, x.im)
                }
                KernelInstruction::Parameter(id) => format!("{complex}(p[{}], 0.0)", id.index()),
                KernelInstruction::Unary { op, input } => Self::emit_unary(*op, *input, complex),
                KernelInstruction::Binary { op, lhs, rhs } => {
                    Self::emit_binary(*op, *lhs, *rhs, complex)
                }
                KernelInstruction::Add(ids) => Self::emit_variadic(ids, false)?,
                KernelInstruction::Mul(ids) => Self::emit_variadic(ids, true)?,
                KernelInstruction::Complex { re, im } => {
                    format!("{complex}({}.x, {}.x)", v(*re), v(*im))
                }
                KernelInstruction::Vector(elements) => Self::aggregate(
                    elements.iter().map(|element| v(*element)),
                    elements.len(),
                    precision,
                ),
                KernelInstruction::Matrix { elements, .. } => Self::aggregate(
                    elements.iter().map(|element| v(*element)),
                    elements.len(),
                    precision,
                ),
                KernelInstruction::Component { input, index } => {
                    format!("{}[{index}]", v(*input))
                }
                KernelInstruction::MatrixElement { input, row, col } => {
                    let KernelValueKind::Matrix { cols, .. } = values[input.index()].kind else {
                        unreachable!("matrix-element IR was validated")
                    };
                    format!("{}[{}]", v(*input), row * cols + col)
                }
                KernelInstruction::Dot { lhs, rhs } => Self::emit_dot(values, *lhs, *rhs)?,
                KernelInstruction::MatVec { matrix, vector } => {
                    Self::emit_matrix_vector(values, *matrix, *vector, precision)?
                }
                KernelInstruction::MatMul { lhs, rhs } => {
                    Self::emit_matrix_product(values, *lhs, *rhs, precision)?
                }
                instruction => {
                    return Err(WgpuError::UnsupportedInstruction(format!(
                        "{instruction:?}"
                    )));
                }
            };
            source.push_str(&format!("let v{index} = {expr};\n"));
        }
        Ok(source)
    }

    fn emit_solve(
        index: usize,
        matrix: &str,
        rhs: &str,
        dimension: usize,
        precision: crate::WgpuPrecision,
    ) -> String {
        let mut source = format!(
            "var lu{index} = {matrix};\nvar x{index} = {rhs};\nvar piv{index}: array<u32, {dimension}>;\n"
        );
        for row in 0..dimension {
            source.push_str(&format!("piv{index}[{row}] = {row}u;\n"));
        }
        source.push_str(&format!(
            "for (var k{index}=0u; k{index}<{dimension}u; k{index}++) {{\n\
var best{index}=k{index};\nvar best_norm{index}=cnorm(lu{index}[k{index}*{dimension}u+k{index}]);\n\
for (var r{index}=k{index}+1u; r{index}<{dimension}u; r{index}++) {{ let candidate=cnorm(lu{index}[r{index}*{dimension}u+k{index}]); if (candidate > best_norm{index}) {{ best_norm{index}=candidate; best{index}=r{index}; }} }}\n\
if (best{index} != k{index}) {{ for (var c{index}=0u; c{index}<{dimension}u; c{index}++) {{ let swap=lu{index}[k{index}*{dimension}u+c{index}]; lu{index}[k{index}*{dimension}u+c{index}]=lu{index}[best{index}*{dimension}u+c{index}]; lu{index}[best{index}*{dimension}u+c{index}]=swap; }} let ps=piv{index}[k{index}]; piv{index}[k{index}]=piv{index}[best{index}]; piv{index}[best{index}]=ps; }}\nif (!(best_norm{index} > 0.0)) {{ atomicMin(&solve_error[0], row); lu{index}[k{index}*{dimension}u+k{index}]=vec2(1.0, 0.0); }}\n\
for (var r{index}=k{index}+1u; r{index}<{dimension}u; r{index}++) {{ let factor=cdiv(lu{index}[r{index}*{dimension}u+k{index}], lu{index}[k{index}*{dimension}u+k{index}]); lu{index}[r{index}*{dimension}u+k{index}]=factor; for (var c{index}=k{index}+1u; c{index}<{dimension}u; c{index}++) {{ lu{index}[r{index}*{dimension}u+c{index}] -= cmul(factor, lu{index}[k{index}*{dimension}u+c{index}]); }} }}\n}}\n"
        ));
        source.push_str(&format!(
            "var y{index}: array<{complex}, {dimension}>;\nfor (var i{index}=0u; i{index}<{dimension}u; i{index}++) {{ var sum=x{index}[piv{index}[i{index}]]; for (var j{index}=0u; j{index}<i{index}; j{index}++) {{ sum -= cmul(lu{index}[i{index}*{dimension}u+j{index}], y{index}[j{index}]); }} y{index}[i{index}]=sum; }}\n\
for (var ri{index}=0u; ri{index}<{dimension}u; ri{index}++) {{ let i={dimension}u-1u-ri{index}; var sum=y{index}[i]; for (var j{index}=i+1u; j{index}<{dimension}u; j{index}++) {{ sum -= cmul(lu{index}[i*{dimension}u+j{index}], x{index}[j{index}]); }} x{index}[i]=cdiv(sum, lu{index}[i*{dimension}u+i]); }}\nlet v{index}=x{index};\n",
            complex = WgslDialect::for_precision(precision)
                .expect("kernel precision is resolved")
                .complex_type(),
        ));
        source
    }

    pub(crate) fn cache_wgsl(
        ir: &CacheKernelIr,
        input_slots: usize,
        layout: &CacheLayout,
        precision: crate::WgpuPrecision,
    ) -> WgpuResult<String> {
        if ir.outputs().len() != layout.offsets().len() {
            return Err(WgpuError::UnsupportedInstruction(
                "cache IR and cache layout have different slot counts".into(),
            ));
        }
        let output_width = layout.width();
        let mut source = format!(
            "@group(0) @binding({parameters}) var<storage, read> inputs: array<vec2<{scalar}>>;\n@group(0) @binding({cache}) var<storage, read_write> cache: array<vec2<{scalar}>>;\n@group(0) @binding({solve_error}) var<storage, read_write> solve_error: array<atomic<u32>>;\n{}@compute @workgroup_size({workgroup}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\nlet row=gid.x;\nif (row >= arrayLength(&cache)/{output_width}u) {{ return; }}\n",
            Self::scalar_prelude(precision),
            scalar = WgslDialect::for_precision(precision)
                .expect("kernel precision is resolved")
                .scalar_type(),
            parameters = Binding::Parameters.index(),
            cache = Binding::Cache.index(),
            solve_error = Binding::SolveError.index(),
            workgroup = WORKGROUP_SIZE,
        );
        source.push_str(&Self::emit_values(ir.values(), precision, |slot, _| {
            format!("inputs[row * {input_slots}u + {slot}u]")
        })?);
        for (slot, output) in ir.outputs().iter().enumerate() {
            let width = ir.values()[output.index()].kind.width();
            for element in 0..width {
                let value = if width == 1 {
                    format!("v{}", output.index())
                } else {
                    format!("v{}[{element}]", output.index())
                };
                source.push_str(&format!(
                    "cache[row * {output_width}u + {}u] = {value};\n",
                    layout.offsets()[slot] + element
                ));
            }
        }
        source.push_str("}\n");
        Ok(source)
    }

    pub(crate) fn gradient_wgsl(
        ir: &GradientKernelIr,
        layout: &CacheLayout,
        precision: crate::WgpuPrecision,
    ) -> WgpuResult<String> {
        let width = ir.outputs().len() + 1;
        let mut source = format!(
            "@group(0) @binding({parameters}) var<storage, read> p: array<{scalar}>;\n@group(0) @binding({cache}) var<storage, read> cache: array<vec2<{scalar}>>;\n@group(0) @binding({weights}) var<storage, read> weights: array<{scalar}>;\n@group(0) @binding({config}) var<storage, read> config: array<u32>;\n@group(0) @binding({partials}) var<storage, read_write> partials: array<{scalar}>;\n@group(0) @binding({reduction_error}) var<storage, read_write> reduction_error: array<atomic<u32>>;\n@group(0) @binding({solve_error}) var<storage, read_write> solve_error: array<atomic<u32>>;\nvar<workgroup> sums: array<{scalar}, {workgroup}>;\n{}fn model_gradient(row: u32) -> array<{scalar}, {width}> {{\n",
            Self::scalar_prelude(precision),
            scalar = WgslDialect::for_precision(precision)
                .expect("kernel precision is resolved")
                .scalar_type(),
            parameters = Binding::Parameters.index(),
            cache = Binding::Cache.index(),
            weights = Binding::Weights.index(),
            config = Binding::Config.index(),
            partials = Binding::Partials.index(),
            reduction_error = Binding::ReductionError.index(),
            solve_error = Binding::SolveError.index(),
            workgroup = WORKGROUP_SIZE,
        );
        source.push_str(&Self::emit_values(ir.values(), precision, |slot, kind| {
            let offset = layout.offsets()[slot];
            if kind.width() == 1 {
                format!("cache[row * {}u + {offset}u]", layout.width())
            } else {
                Self::aggregate(
                    (0..kind.width()).map(|element| {
                        format!("cache[row * {}u + {}u]", layout.width(), offset + element)
                    }),
                    kind.width(),
                    precision,
                )
            }
        })?);
        let outputs = std::iter::once(format!("v{}.x", ir.primal_root().index()))
            .chain(
                ir.outputs()
                    .iter()
                    .map(|output| format!("v{}.x", output.index())),
            )
            .collect::<Vec<_>>()
            .join(", ");
        source.push_str(&format!(
            "return array<{scalar}, {width}>({outputs});\n}}\n@compute @workgroup_size({workgroup}) fn reduce_gradient(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\nvar result: array<{scalar}, {width}>;\nvar scale: {scalar}=0.0;\nif (gid.x < arrayLength(&weights)) {{ result=model_gradient(gid.x); let value=result[0]; if (config[0] == 0u) {{ scale=1.0; }} else if (value <= 0.0) {{ atomicMin(&reduction_error[0], gid.x); }} else if (config[0] == 1u) {{ scale=1.0; }} else {{ scale=1.0/value; }} result[0]=select(value, scalar_log(value), config[0] == 2u); scale *= weights[gid.x]; result[0] *= weights[gid.x]; }}\nfor (var component=0u; component<{width}u; component++) {{ if (component == 0u) {{ sums[lid.x]=result[0]; }} else {{ sums[lid.x]=result[component]*scale; }} workgroupBarrier(); var stride=32u; loop {{ if (lid.x < stride) {{ sums[lid.x] += sums[lid.x+stride]; }} workgroupBarrier(); if (stride == 1u) {{ break; }} stride/=2u; }} if (lid.x == 0u) {{ partials[wid.x*{width}u+component]=sums[0]; }} workgroupBarrier(); }}\n}}\n",
            scalar = WgslDialect::for_precision(precision)
                .expect("kernel precision is resolved")
                .scalar_type(),
            workgroup = WORKGROUP_SIZE
        ));
        Ok(source)
    }

    pub(crate) fn wgsl(
        ir: &ScalarKernelIr,
        layout: &CacheLayout,
        precision: crate::WgpuPrecision,
    ) -> WgpuResult<String> {
        let mut source = format!(
            "@group(0) @binding({parameters}) var<storage, read> p: array<{scalar}>;\n@group(0) @binding({cache}) var<storage, read> cache: array<vec2<{scalar}>>;\n@group(0) @binding({output}) var<storage, read_write> out: array<vec2<{scalar}>>;\n@group(0) @binding({weights}) var<storage, read> weights: array<{scalar}>;\n@group(0) @binding({config}) var<storage, read> config: array<u32>;\n@group(0) @binding({partials}) var<storage, read_write> partials: array<{scalar}>;\n@group(0) @binding({reduction_error}) var<storage, read_write> reduction_error: array<atomic<u32>>;\n@group(0) @binding({solve_error}) var<storage, read_write> solve_error: array<atomic<u32>>;\nvar<workgroup> sums: array<{scalar}, {workgroup}>;\n{}fn model(row: u32) -> vec2<{scalar}> {{\n",
            Self::scalar_prelude(precision),
            scalar = WgslDialect::for_precision(precision)
                .expect("kernel precision is resolved")
                .scalar_type(),
            parameters = Binding::Parameters.index(),
            cache = Binding::Cache.index(),
            output = Binding::Output.index(),
            weights = Binding::Weights.index(),
            config = Binding::Config.index(),
            partials = Binding::Partials.index(),
            reduction_error = Binding::ReductionError.index(),
            solve_error = Binding::SolveError.index(),
            workgroup = WORKGROUP_SIZE
        );
        source.push_str(&Self::emit_values(ir.values(), precision, |slot, kind| {
            let offset = layout.offsets()[slot];
            if kind.width() == 1 {
                format!("cache[row * {}u + {offset}u]", layout.width())
            } else {
                Self::aggregate(
                    (0..kind.width()).map(|element| {
                        format!("cache[row * {}u + {}u]", layout.width(), offset + element)
                    }),
                    kind.width(),
                    precision,
                )
            }
        })?);
        let v = |id: KernelValueId| format!("v{}", id.index());
        source.push_str(&format!(
            "return {};\n}}\n@compute @workgroup_size({workgroup}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\nif (gid.x >= arrayLength(&out)) {{ return; }}\nout[gid.x] = model(gid.x);\n}}\n@compute @workgroup_size({workgroup}) fn reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\nvar contribution: {scalar} = 0.0;\nif (gid.x < arrayLength(&weights)) {{\nlet value = model(gid.x).x;\nif (config[0] == 0u) {{ contribution = value; }} else if (value <= 0.0) {{ atomicMin(&reduction_error[0], gid.x); }} else if (config[0] == 1u) {{ contribution = value; }} else {{ contribution = scalar_log(value); }}\ncontribution *= weights[gid.x];\n}}\nsums[lid.x] = contribution;\nworkgroupBarrier();\nvar stride = 32u;\nloop {{\nif (lid.x < stride) {{ sums[lid.x] += sums[lid.x + stride]; }}\nworkgroupBarrier();\nif (stride == 1u) {{ break; }}\nstride /= 2u;\n}}\nif (lid.x == 0u) {{ partials[wid.x] = sums[0]; }}\n}}\n",
            v(ir.root()),
            scalar = WgslDialect::for_precision(precision)
                .expect("kernel precision is resolved")
                .scalar_type(),
            workgroup = WORKGROUP_SIZE
        ));
        Ok(source)
    }
}

#[cfg(test)]
mod tests {
    use super::{ScalarType, WgslDialect};

    #[test]
    fn dialect_exposes_structural_target_vocabulary() {
        let dialect = WgslDialect {
            scalar: ScalarType::F64,
        };
        assert_eq!(dialect.scalar_type(), "f64");
        assert_eq!(dialect.complex_type(), "vec2<f64>");
    }

    #[test]
    fn dialect_exposes_target_types() {
        let dialect = WgslDialect {
            scalar: ScalarType::F32,
        };
        assert_eq!(dialect.scalar_type(), "f32");
    }
}
