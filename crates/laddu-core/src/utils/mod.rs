use std::marker::PhantomData;

use nalgebra::SMatrix;
use num::complex::Complex64;
use polars::prelude::*;

/// Useful enumerations for various frames and variables common in particle physics analyses.
pub mod enums;
/// Standard special functions like spherical harmonics and momentum definitions.
pub mod functions;
/// Traits and structs which can be used to extract complex information from
/// [`Event`](crate::data::Event)s.
pub mod variables;
/// Traits to give additional functionality to [`nalgebra::Vector3`] and [`nalgebra::Vector4`] (in
/// particular, to treat the latter as a four-momentum).
pub mod vectors;

/// A helper method to get histogram edges from evenly-spaced `bins` over a given `range`
/// # See Also
/// [`Histogram`]
/// [`get_bin_index`]
pub fn get_bin_edges(bins: usize, range: (f64, f64)) -> Vec<f64> {
    let bin_width = (range.1 - range.0) / (bins as f64);
    (0..=bins)
        .map(|i| range.0 + (i as f64 * bin_width))
        .collect()
}

/// A helper method to obtain the index of a bin where a value should go in a histogram with evenly
/// spaced `bins` over a given `range`
///
/// # See Also
/// [`Histogram`]
/// [`get_bin_edges`]
pub fn get_bin_index(value: f64, bins: usize, limits: (f64, f64)) -> Option<usize> {
    if value >= limits.0 && value < limits.1 {
        let bin_width = (limits.1 - limits.0) / bins as f64;
        let bin_index = ((value - limits.0) / bin_width).floor() as usize;
        Some(bin_index.min(bins - 1))
    } else {
        None
    }
}

pub fn get_bin_index_polars(expr: Expr, bins: usize, limits: (f64, f64)) -> Expr {
    let bin_width = (limits.1 - limits.0) / bins as f64;
    when(
        expr.clone()
            .lt(lit(limits.0))
            .or(expr.clone().gt_eq(lit(limits.1))),
    )
    .then(lit(bins as u64))
    .otherwise(
        ((expr - lit(limits.0)) / lit(bin_width))
            .floor()
            .cast(DataType::UInt64),
    )
}

/// A simple struct which represents a histogram
pub struct Histogram {
    /// The number of counts in each bin (can be `f64`s since these might be weighted counts)
    pub counts: Vec<f64>,
    /// The edges of each bin (length is one greater than `counts`)
    pub bin_edges: Vec<f64>,
}

/// A method which creates a histogram from some data by binning it with evenly spaced `bins` within
/// the given `range`
pub fn histogram<T: AsRef<[f64]>>(
    values: T,
    bins: usize,
    range: (f64, f64),
    weights: Option<T>,
) -> Histogram {
    assert!(bins > 0, "Number of bins must be greater than zero!");
    assert!(
        range.1 > range.0,
        "The lower edge of the range must be smaller than the upper edge!"
    );
    if let Some(w) = &weights {
        assert_eq!(
            values.as_ref().len(),
            w.as_ref().len(),
            "`values` and `weights` must have the same length!"
        );
    }
    let mut counts = vec![0.0; bins];
    for (i, &value) in values.as_ref().iter().enumerate() {
        if let Some(bin_index) = get_bin_index(value, bins, range) {
            let weight = weights.as_ref().map_or(1.0, |w| w.as_ref()[i]);
            counts[bin_index] += weight;
        }
    }
    Histogram {
        counts,
        bin_edges: get_bin_edges(bins, range),
    }
}

const RE_SUFFIX: &str = " real";
const IM_SUFFIX: &str = " imag";
#[inline]
fn pack_struct_many(base: &str, fields: Vec<Series>) -> PolarsResult<Column> {
    debug_assert!(!fields.is_empty());
    let sc = StructChunked::from_series(base.into(), fields[0].len(), fields.iter())?;
    Ok(sc.into_series().into())
}

pub trait FlattenF64 {
    /// number of f64 lanes produced
    const NCOMP: usize;
    /// write flattened components (re/im interleaved for complex) into dst
    fn write_flattened(&self, dst: &mut [f64]);
    /// field names for the struct columns
    fn field_names(base: &str) -> Vec<String>;
}

impl FlattenF64 for f64 {
    const NCOMP: usize = 1;
    fn write_flattened(&self, dst: &mut [f64]) {
        dst[0] = *self;
    }
    fn field_names(base: &str) -> Vec<String> {
        vec![base.to_string()]
    }
}
impl FlattenF64 for Complex64 {
    const NCOMP: usize = 2;
    fn write_flattened(&self, dst: &mut [f64]) {
        dst[0] = self.re;
        dst[1] = self.im;
    }
    fn field_names(base: &str) -> Vec<String> {
        vec![format!("{base}{RE_SUFFIX}"), format!("{base}{IM_SUFFIX}")]
    }
}
impl<const R: usize, const C: usize> FlattenF64 for SMatrix<f64, R, C> {
    const NCOMP: usize = R * C;

    fn write_flattened(&self, dst: &mut [f64]) {
        // col-major
        for c in 0..C {
            for r in 0..R {
                dst[c * R + r] = self[(r, c)];
            }
        }
    }

    fn field_names(base: &str) -> Vec<String> {
        if C == 1 {
            // vector naming: base_0, base_1, ...
            (0..R).map(|r| format!("{base}_{r}")).collect()
        } else {
            // matrix naming: base_r_c
            let mut names = Vec::with_capacity(R * C);
            for c in 0..C {
                for r in 0..R {
                    names.push(format!("{base}_{r}_{c}"));
                }
            }
            names
        }
    }
}

// Complex matrix or vector (C may be 1)
impl<const R: usize, const C: usize> FlattenF64 for SMatrix<Complex64, R, C> {
    const NCOMP: usize = 2 * R * C;

    fn write_flattened(&self, dst: &mut [f64]) {
        // col-major, re/imag interleaved per entry
        for c in 0..C {
            for r in 0..R {
                let z = self[(r, c)];
                let k = (c * R + r) * 2;
                dst[k] = z.re;
                dst[k + 1] = z.im;
            }
        }
    }

    fn field_names(base: &str) -> Vec<String> {
        let mut names = Vec::with_capacity(2 * R * C);
        if C == 1 {
            // vector naming: base_0_re, base_0_im, base_1_re, ...
            for r in 0..R {
                names.push(format!("{base}_{r}{RE_SUFFIX}"));
                names.push(format!("{base}_{r}{IM_SUFFIX}"));
            }
        } else {
            // matrix naming: base_r_c_re, base_r_c_im
            for c in 0..C {
                for r in 0..R {
                    names.push(format!("{base}_{r}_{c}{RE_SUFFIX}"));
                    names.push(format!("{base}_{r}_{c}{IM_SUFFIX}"));
                }
            }
        }
        names
    }
}
pub struct Vectorized<O: FlattenF64> {
    base: String,
    struct_expr: Expr,
    _m: PhantomData<O>,
}

impl<O: FlattenF64> Clone for Vectorized<O> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            struct_expr: self.struct_expr.clone(),
            _m: PhantomData,
        }
    }
}

impl<O: FlattenF64> Vectorized<O> {
    fn field_exprs(&self) -> Vec<Expr> {
        O::field_names(&self.base)
            .into_iter()
            .map(|n| self.struct_expr.clone().struct_().field_by_name(&n))
            .collect()
    }

    pub fn as_scalar(self) -> Expr {
        debug_assert!(O::NCOMP == 1);
        self.field_exprs().into_iter().next().unwrap()
    }
    pub fn as_cscalar(self) -> (Expr, Expr) {
        debug_assert!(O::NCOMP == 2);
        let v = self.field_exprs();
        (v[0].clone(), v[1].clone())
    }
    pub fn as_vector<const N: usize>(self) -> [Expr; N] {
        debug_assert!(O::NCOMP == N);
        self.field_exprs().try_into().ok().unwrap()
    }
    pub fn as_cvector<const N: usize>(self) -> [(Expr, Expr); N] {
        debug_assert!(O::NCOMP == 2 * N);
        let v = self.field_exprs();
        let mut out: Vec<(Expr, Expr)> = Vec::with_capacity(N);
        for i in 0..N {
            out.push((v[2 * i].clone(), v[2 * i + 1].clone()));
        }
        out.try_into().ok().unwrap()
    }
    pub fn as_matrix<const R: usize, const C: usize>(self) -> [[Expr; C]; R] {
        debug_assert!(O::NCOMP == R * C);
        let v = self.field_exprs();
        core::array::from_fn(|r| core::array::from_fn(|c| v[c * R + r].clone()))
    }
    pub fn as_cmatrix<const R: usize, const C: usize>(self) -> [[(Expr, Expr); C]; R] {
        debug_assert!(O::NCOMP == 2 * R * C);
        let v = self.field_exprs();
        core::array::from_fn(|r| {
            core::array::from_fn(|c| {
                let k = (c * R + r) * 2;
                (v[k].clone(), v[k + 1].clone())
            })
        })
    }
}
/// Vectorize a function `f: (&[f64]) -> O` over rows of the given input Exprs.
/// Any extra non-column args can be captured in the closure.
pub fn vectorize<O, F, const K: usize>(
    base: impl Into<String>,
    inputs: [Expr; K],
    f: F,
) -> Vectorized<O>
where
    O: FlattenF64 + Send + Sync + 'static,
    F: Fn(&[f64]) -> O + Send + Sync + Clone + 'static,
{
    let base = base.into();
    let base_name = base.clone();
    let base_schema = base.clone();
    let field_names = O::field_names(&base);
    let f_closure = f.clone();

    let struct_expr = map_multiple(
        move |cols: &mut [Column]| {
            // pull contiguous f64 slices, nulls not supported
            let mut slices: Vec<&[f64]> = Vec::with_capacity(K);
            for j in 0..K {
                let ca = cols[j].f64()?;
                if ca.null_count() != 0 {
                    polars_bail!(ComputeError: "vectorize({base_name}): nulls not supported");
                }
                slices.push(ca.cont_slice()?);
            }
            let n = slices[0].len();

            // component buffers
            let mut comps: Vec<Vec<f64>> = (0..O::NCOMP).map(|_| Vec::with_capacity(n)).collect();

            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                let rows: Vec<Vec<f64>> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let mut row = Vec::with_capacity(K);
                        for j in 0..K {
                            row.push(slices[j][i]);
                        }
                        row
                    })
                    .collect();

                // compute then scatter into component buffers (serial scatter to keep order)
                let flat_per_row: Vec<Vec<f64>> = rows
                    .par_iter()
                    .map(|row| {
                        let out = f_closure(row);
                        let mut tmp = vec![0.0f64; O::NCOMP];
                        out.write_flattened(&mut tmp);
                        tmp
                    })
                    .collect();

                for tmp in flat_per_row {
                    for k in 0..O::NCOMP {
                        comps[k].push(tmp[k]);
                    }
                }
            }
            #[cfg(not(feature = "rayon"))]
            {
                let rows: Vec<Vec<f64>> = (0..n)
                    .into_iter()
                    .map(|i| {
                        let mut row = Vec::with_capacity(K);
                        for j in 0..K {
                            row.push(slices[j][i]);
                        }
                        row
                    })
                    .collect();

                // compute then scatter into component buffers (serial scatter to keep order)
                let flat_per_row: Vec<Vec<f64>> = rows
                    .iter()
                    .map(|row| {
                        let out = f_closure(row);
                        let mut tmp = vec![0.0f64; O::NCOMP];
                        out.write_flattened(&mut tmp);
                        tmp
                    })
                    .collect();

                for tmp in flat_per_row {
                    for k in 0..O::NCOMP {
                        comps[k].push(tmp[k]);
                    }
                }
            }

            // build struct of Float64 fields
            let fields: Vec<Series> = field_names
                .iter()
                .enumerate()
                .map(|(k, name)| {
                    Float64Chunked::from_vec(name.clone().into(), std::mem::take(&mut comps[k]))
                        .into_series()
                })
                .collect();

            pack_struct_many(&base_name, fields)
        },
        inputs,
        move |_schema, _inputs| {
            let fields = O::field_names(&base_schema)
                .into_iter()
                .map(|n| Field::new(n.into(), DataType::Float64))
                .collect::<Vec<_>>();
            Ok(Field::new(
                base_schema.clone().into(),
                DataType::Struct(fields),
            ))
        },
    )
    .alias(&base);

    Vectorized {
        base,
        struct_expr,
        _m: PhantomData,
    }
}

pub type CExpr = (Expr, Expr); // (re, im)

pub trait ComplexExprExt {
    #[inline]
    fn complex(re: Expr, im: Expr) -> CExpr
    where
        Self: Sized,
    {
        (re, im)
    }

    fn from_phase(phi: Expr) -> CExpr
    where
        Self: Sized,
    {
        (phi.clone().cos(), phi.sin())
    }

    fn real(&self) -> Expr;
    fn imag(&self) -> Expr;
    fn phase(&self) -> Expr; // arg(z) = atan2(im, re)
    fn conj(&self) -> CExpr;

    fn add_complex(&self, rhs: &CExpr) -> CExpr;
    fn sub_complex(&self, rhs: &CExpr) -> CExpr;
    fn mul_complex(&self, rhs: &CExpr) -> CExpr;
    fn div_complex(&self, rhs: &CExpr) -> CExpr;

    fn add_scalar(&self, rhs: &Expr) -> CExpr;
    fn sub_scalar(&self, rhs: &Expr) -> CExpr;
    fn rsub_scalar(&self, rhs: &Expr) -> CExpr;

    fn mul_scalar(&self, rhs: &Expr) -> CExpr;
    fn div_scalar(&self, rhs: &Expr) -> CExpr;
    fn rdiv_scalar(&self, rhs: &Expr) -> CExpr;
}

impl ComplexExprExt for CExpr {
    #[inline]
    fn real(&self) -> Expr {
        self.0.clone()
    }
    #[inline]
    fn imag(&self) -> Expr {
        self.1.clone()
    }

    #[inline]
    fn phase(&self) -> Expr {
        self.1.clone().arctan2(self.0.clone())
    }

    #[inline]
    fn conj(&self) -> CExpr {
        (self.0.clone(), -self.1.clone())
    }

    #[inline]
    fn add_complex(&self, rhs: &CExpr) -> CExpr {
        (
            self.0.clone() + rhs.0.clone(),
            self.1.clone() + rhs.1.clone(),
        )
    }
    #[inline]
    fn sub_complex(&self, rhs: &CExpr) -> CExpr {
        (
            self.0.clone() - rhs.0.clone(),
            self.1.clone() - rhs.1.clone(),
        )
    }
    #[inline]
    fn mul_complex(&self, rhs: &CExpr) -> CExpr {
        let (ar, ai) = (self.0.clone(), self.1.clone());
        let (br, bi) = (rhs.0.clone(), rhs.1.clone());
        (
            ar.clone() * br.clone() - ai.clone() * bi.clone(),
            ar * bi + ai * br,
        )
    }
    #[inline]
    fn div_complex(&self, rhs: &CExpr) -> CExpr {
        let (ar, ai) = (self.0.clone(), self.1.clone());
        let (br, bi) = (rhs.0.clone(), rhs.1.clone());
        let denom = br.clone() * br.clone() + bi.clone() * bi.clone();
        (
            (ar.clone() * br.clone() + ai.clone() * bi.clone()) / denom.clone(),
            (ai * br - ar * bi) / denom,
        )
    }

    #[inline]
    fn add_scalar(&self, rhs: &Expr) -> CExpr {
        (self.0.clone() + rhs.clone(), self.1.clone())
    }
    #[inline]
    fn sub_scalar(&self, rhs: &Expr) -> CExpr {
        (self.0.clone() - rhs.clone(), self.1.clone())
    }
    #[inline]
    fn rsub_scalar(&self, rhs: &Expr) -> CExpr {
        (rhs.clone() - self.0.clone(), -self.1.clone())
    }
    #[inline]
    fn mul_scalar(&self, rhs: &Expr) -> CExpr {
        (self.0.clone() * rhs.clone(), self.1.clone() * rhs.clone())
    }
    #[inline]
    fn div_scalar(&self, rhs: &Expr) -> CExpr {
        (self.0.clone() / rhs.clone(), self.1.clone() / rhs.clone())
    }
    #[inline]
    fn rdiv_scalar(&self, rhs: &Expr) -> CExpr {
        // rhs / (a+ib) = rhs*(a-ib)/(a^2+b^2)
        let (ar, ai) = (self.0.clone(), self.1.clone());
        let denom = ar.clone() * ar.clone() + ai.clone() * ai.clone();
        (
            (rhs.clone() * ar.clone()) / denom.clone(),
            (-rhs.clone() * ai) / denom,
        )
    }
}

#[inline]
fn list_to_name<I, S>(values: &I) -> String
where
    I: IntoIterator<Item = S> + Clone,
    S: Into<PlSmallStr>,
{
    values
        .clone()
        .into_iter()
        .map(|s| s.into().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    // use std::sync::Arc;

    // use crate::{
    //     data::test_dataset,
    //     traits::Variable,
    //     utils::{get_bin_index, histogram},
    //     Mass,
    // };
    //
    // #[test]
    // fn test_binning() {
    //     let v = Mass::new([2]);
    //     let dataset = Arc::new(test_dataset());
    //     let bin_index = get_bin_index(v.value_on(&dataset)[0], 3, (0.0, 1.0));
    //     assert_eq!(bin_index, Some(1));
    //     let bin_index = get_bin_index(0.0, 3, (0.0, 1.0));
    //     assert_eq!(bin_index, Some(0));
    //     let bin_index = get_bin_index(0.1, 3, (0.0, 1.0));
    //     assert_eq!(bin_index, Some(0));
    //     let bin_index = get_bin_index(0.9, 3, (0.0, 1.0));
    //     assert_eq!(bin_index, Some(2));
    //     let bin_index = get_bin_index(1.0, 3, (0.0, 1.0));
    //     assert_eq!(bin_index, None);
    //     let bin_index = get_bin_index(2.0, 3, (0.0, 1.0));
    //     assert_eq!(bin_index, None);
    //     let histogram = histogram(v.value_on(&dataset), 3, (0.0, 1.0), Some(dataset.weights()));
    //     assert_eq!(histogram.counts, vec![0.0, 0.48, 0.0]);
    //     assert_eq!(histogram.bin_edges, vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    // }
}
