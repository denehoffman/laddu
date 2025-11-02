use crate::{
    amplitudes::{AmplitudeID, ParameterLike},
    LadduError, LadduResult,
};
use indexmap::IndexSet;
use nalgebra::{SMatrixView, SVectorView};
use num::complex::Complex64;
use polars::prelude::*;
use polars_arrow::array::{Array, FixedSizeListArray, PrimitiveArray};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "rayon")]
use rayon::{iter::ParallelIterator, prelude::*};

/// This struct holds references to the constants and free parameters used in the fit so that they
/// may be obtained from their corresponding [`ParameterID`].
#[derive(Debug)]
pub struct Parameters<'a> {
    pub(crate) parameters: &'a [f64],
    pub(crate) constants: &'a [f64],
}

impl<'a> Parameters<'a> {
    /// Create a new set of [`Parameters`] from a list of floating values and a list of constant values
    pub fn new(parameters: &'a [f64], constants: &'a [f64]) -> Self {
        Self {
            parameters,
            constants,
        }
    }

    /// Obtain a parameter value or constant value from the given [`ParameterID`].
    pub fn get(&self, pid: ParameterID) -> f64 {
        match pid {
            ParameterID::Parameter(index) => self.parameters[index],
            ParameterID::Constant(index) => self.constants[index],
            ParameterID::Uninit => panic!("Parameter has not been registered!"),
        }
    }

    /// The number of free parameters.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.parameters.len()
    }
}

/// An object which acts as a tag to refer to either a free parameter or a constant value.
#[derive(Default, Copy, Clone, Debug, Serialize, Deserialize)]
pub enum ParameterID {
    /// A free parameter.
    Parameter(usize),
    /// A constant value.
    Constant(usize),
    /// An uninitialized ID
    #[default]
    Uninit,
}

#[inline]
fn f64s_as_c64s(flat: &[f64]) -> &[Complex64] {
    debug_assert!(flat.len() % 2 == 0);
    unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const Complex64, flat.len() / 2) }
}

fn make_array(exprs: Vec<Expr>, width: usize) -> ColumnSpec {
    let expr = concat_list(exprs)
        .expect("concat_list failed")
        .cast(DataType::Array(Box::new(DataType::Float64), width));
    ColumnSpec { expr, width }
}

pub struct CacheRows {
    df: DataFrame,
    names: Vec<String>,
    widths: Vec<usize>,
    rows: usize,
}
impl CacheRows {
    pub fn new(df: DataFrame, names: Vec<String>, widths: Vec<usize>) -> PolarsResult<Self> {
        Ok(Self {
            rows: df.height(),
            df,
            names,
            widths,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn row(&self, i: usize) -> CacheRow<'_> {
        CacheRow { ctx: self, i }
    }
    pub fn dataframe(&self) -> &DataFrame {
        &self.df
    }

    pub fn iter(&self) -> impl Iterator<Item = CacheRow<'_>> {
        (0..self.rows).map(move |i| CacheRow { ctx: self, i })
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = CacheRow<'_>> {
        (0..self.rows)
            .into_par_iter()
            .map(move |i| CacheRow { ctx: self, i })
    }
}

impl<'a> IntoIterator for &'a CacheRows {
    type Item = CacheRow<'a>;

    type IntoIter = CacheRowsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        CacheRowsIter {
            ctx: self,
            idx: 0,
            nrows: self.df.height(),
        }
    }
}

pub struct CacheRowsIter<'a> {
    ctx: &'a CacheRows,
    idx: usize,
    nrows: usize,
}

impl<'a> Iterator for CacheRowsIter<'a> {
    type Item = CacheRow<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.nrows {
            let row = CacheRow {
                ctx: self.ctx,
                i: self.idx,
            };
            self.idx += 1;
            Some(row)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.nrows - self.idx;
        (remaining, Some(remaining))
    }
}

pub struct CacheRow<'a> {
    ctx: &'a CacheRows,
    i: usize,
}
impl<'a> CacheRow<'a> {
    #[inline]
    fn series_and_width(&'a self, k: ExprID) -> (&'a Series, usize) {
        let id = k.get_id();
        let name = &self.ctx.names[id];
        let col = self.ctx.df.column(name).expect("missing column"); // &Column
        let s: &'a Series = col.as_materialized_series(); // &Series
        (s, self.ctx.widths[id])
    }
    #[inline]
    fn row_slice<'s>(s: &'s Series, width: usize, row: usize) -> (&'s [f64], usize) {
        let ac = s.array().expect("not Array dtype");
        let arr = ac.downcast_iter().next().expect("empty chunked array");
        let fis = arr
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("not FixedSizeList");
        debug_assert_eq!(fis.size() as usize, width);
        let child = fis
            .values()
            .as_any()
            .downcast_ref::<PrimitiveArray<f64>>()
            .expect("child not f64");
        let vals: &[f64] = child.values().as_slice();
        let start = row * width;
        (&vals[start..start + width], width)
    }

    pub fn get_weight(&self) -> f64 {
        let (s, w) = self.series_and_width(ExprID::Expr(0));
        debug_assert_eq!(w, 1);
        Self::row_slice(s, w, self.i).0[0]
    }

    pub fn get_scalar(&self, k: ExprID) -> f64 {
        let (s, w) = self.series_and_width(k);
        debug_assert_eq!(w, 1);
        Self::row_slice(s, w, self.i).0[0]
    }
    pub fn get_vector<const N: usize>(&self, k: ExprID) -> SVectorView<'_, f64, N> {
        let (s, w) = self.series_and_width(k);
        debug_assert_eq!(w, N);
        SVectorView::from_slice(Self::row_slice(s, w, self.i).0)
    }
    pub fn get_matrix<const R: usize, const C: usize>(
        &self,
        k: ExprID,
    ) -> SMatrixView<'_, f64, R, C> {
        let (s, w) = self.series_and_width(k);
        debug_assert_eq!(w, R * C);
        SMatrixView::from_slice(Self::row_slice(s, w, self.i).0)
    }

    pub fn get_cscalar(&self, k: ExprID) -> Complex64 {
        let (s, w) = self.series_and_width(k);
        debug_assert_eq!(w, 2);
        let flat = Self::row_slice(s, w, self.i).0;
        Complex64::new(flat[0], flat[1])
    }
    pub fn get_cvector<const N: usize>(&self, k: ExprID) -> SVectorView<'_, Complex64, N> {
        let (s, w) = self.series_and_width(k);
        debug_assert_eq!(w, 2 * N);
        SVectorView::from_slice(f64s_as_c64s(Self::row_slice(s, w, self.i).0))
    }
    pub fn get_cmatrix<const R: usize, const C: usize>(
        &self,
        k: ExprID,
    ) -> SMatrixView<'_, Complex64, R, C> {
        let (s, w) = self.series_and_width(k);
        debug_assert_eq!(w, 2 * R * C);
        SMatrixView::from_slice(f64s_as_c64s(Self::row_slice(s, w, self.i).0))
    }
}

#[derive(Clone)]
pub struct ColumnSpec {
    pub expr: Expr,
    pub width: usize,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Default)]
pub enum ExprID {
    Expr(usize),
    #[default]
    Uninit,
}

impl ExprID {
    #[inline]
    pub fn get_id(&self) -> usize {
        if let Self::Expr(id) = self {
            *id
        } else {
            panic!("Tried to access an uninitialized ExprID!")
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DerivedCol {
    name: String,
    width: usize,
    expr: Expr,
}

/// The main resource manager for cached values, amplitudes, parameters, and constants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resources {
    amplitudes: HashMap<String, AmplitudeID>,
    /// A list indicating which amplitudes are active (using [`AmplitudeID`]s as indices)
    pub active: Vec<bool>,
    /// The set of all registered parameter names across registered [`Amplitude`](`crate::amplitudes::Amplitude`)s
    pub parameters: IndexSet<String>,
    /// Values of all constants across registered [`Amplitude`](`crate::amplitudes::Amplitude`)s
    pub constants: Vec<f64>,
    exprs: Vec<DerivedCol>,
    expr_name_to_id: HashMap<String, ExprID>,
}

impl Default for Resources {
    fn default() -> Self {
        let weight_spec = make_array(vec![col("weight")], 1);
        Self {
            amplitudes: Default::default(),
            active: Default::default(),
            parameters: Default::default(),
            constants: Default::default(),
            exprs: vec![DerivedCol {
                name: "weight".to_string(), // NOTE: no "__" prefix
                width: 1,
                expr: weight_spec.expr,
            }],
            expr_name_to_id: HashMap::from([("weight".to_string(), ExprID::Expr(0))]),
        }
    }
}

#[derive(Default, Clone)]
pub enum ExprName {
    Name(String),
    Infer,
    #[default]
    None,
}
impl From<String> for ExprName {
    fn from(s: String) -> Self {
        Self::Name(s)
    }
}
impl From<&str> for ExprName {
    fn from(value: &str) -> Self {
        Self::Name(value.to_string())
    }
}

impl Resources {
    /// Activate an [`Amplitude`](crate::amplitudes::Amplitude) by name.
    pub fn activate<T: AsRef<str>>(&mut self, name: T) -> Result<(), LadduError> {
        self.active[self
            .amplitudes
            .get(name.as_ref())
            .ok_or(LadduError::AmplitudeNotFoundError {
                name: name.as_ref().to_string(),
            })?
            .1] = true;
        Ok(())
    }
    /// Activate several [`Amplitude`](crate::amplitudes::Amplitude)s by name.
    pub fn activate_many<T: AsRef<str>>(&mut self, names: &[T]) -> Result<(), LadduError> {
        for name in names {
            self.activate(name)?
        }
        Ok(())
    }
    /// Activate all registered [`Amplitude`](crate::amplitudes::Amplitude)s.
    pub fn activate_all(&mut self) {
        self.active = vec![true; self.active.len()];
    }
    /// Deactivate an [`Amplitude`](crate::amplitudes::Amplitude) by name.
    pub fn deactivate<T: AsRef<str>>(&mut self, name: T) -> Result<(), LadduError> {
        self.active[self
            .amplitudes
            .get(name.as_ref())
            .ok_or(LadduError::AmplitudeNotFoundError {
                name: name.as_ref().to_string(),
            })?
            .1] = false;
        Ok(())
    }
    /// Deactivate several [`Amplitude`](crate::amplitudes::Amplitude)s by name.
    pub fn deactivate_many<T: AsRef<str>>(&mut self, names: &[T]) -> Result<(), LadduError> {
        for name in names {
            self.deactivate(name)?;
        }
        Ok(())
    }
    /// Deactivate all registered [`Amplitude`](crate::amplitudes::Amplitude)s.
    pub fn deactivate_all(&mut self) {
        self.active = vec![false; self.active.len()];
    }
    /// Isolate an [`Amplitude`](crate::amplitudes::Amplitude) by name (deactivate the rest).
    pub fn isolate<T: AsRef<str>>(&mut self, name: T) -> Result<(), LadduError> {
        self.deactivate_all();
        self.activate(name)
    }
    /// Isolate several [`Amplitude`](crate::amplitudes::Amplitude)s by name (deactivate the rest).
    pub fn isolate_many<T: AsRef<str>>(&mut self, names: &[T]) -> Result<(), LadduError> {
        self.deactivate_all();
        self.activate_many(names)
    }
    /// Register an [`Amplitude`](crate::amplitudes::Amplitude) with the [`Resources`] manager.
    /// This method should be called at the end of the
    /// [`Amplitude::register`](crate::amplitudes::Amplitude::register) method. The
    /// [`Amplitude`](crate::amplitudes::Amplitude) should probably obtain a name [`String`] in its
    /// constructor.
    ///
    /// # Errors
    ///
    /// The [`Amplitude`](crate::amplitudes::Amplitude)'s name must be unique and not already
    /// registered, else this will return a [`RegistrationError`][LadduError::RegistrationError].
    pub fn register_amplitude(&mut self, name: &str) -> Result<AmplitudeID, LadduError> {
        if self.amplitudes.contains_key(name) {
            return Err(LadduError::RegistrationError {
                name: name.to_string(),
            });
        }
        let next_id = AmplitudeID(name.to_string(), self.amplitudes.len());
        self.amplitudes.insert(name.to_string(), next_id.clone());
        self.active.push(true);
        Ok(next_id)
    }
    /// Register a free parameter (or constant) [`ParameterLike`]. This method should be called
    /// within the [`Amplitude::register`](crate::amplitudes::Amplitude::register) method, and the
    /// resulting [`ParameterID`] should be stored to use later to retrieve the value from the
    /// [`Parameters`] wrapper object.
    pub fn register_parameter(&mut self, pl: &ParameterLike) -> ParameterID {
        match pl {
            ParameterLike::Parameter(name) => {
                let (index, _) = self.parameters.insert_full(name.to_string());
                ParameterID::Parameter(index)
            }
            ParameterLike::Constant(value) => {
                self.constants.push(*value);
                ParameterID::Constant(self.constants.len() - 1)
            }
            ParameterLike::Uninit => panic!("Parameter was not initialized!"),
        }
    }

    pub fn materialize(&self, lf: LazyFrame) -> PolarsResult<CacheRows> {
        let selects: Vec<_> = self
            .exprs
            .iter()
            .map(|d| d.expr.clone().alias(&d.name))
            .collect();

        let mut df = lf.select(selects).collect()?;
        df.rechunk_mut(); // TODO: remove?

        for (i, d) in self.exprs.iter().enumerate() {
            let s = df.column(&d.name)?;
            match s.dtype() {
                DataType::Array(elem, w) if **elem == DataType::Float64 && *w == d.width => {}
                other => {
                    return Err(PolarsError::ComputeError(
                        format!(
                            "Derived col {} has dtype {other:?}, expected Array(Float64,{})",
                            d.name, d.width
                        )
                        .into(),
                    ))
                }
            }
            assert_eq!(Some(i), df.get_column_index(&d.name));
        }

        let names = self.exprs.iter().map(|d| d.name.clone()).collect();
        let widths = self.exprs.iter().map(|d| d.width).collect();
        CacheRows::new(df, names, widths)
    }

    fn register_expr(&mut self, name: ExprName, spec: ColumnSpec) -> LadduResult<ExprID> {
        let (id, col_name) = match name {
            ExprName::Name(name) => (
                *self
                    .expr_name_to_id
                    .entry(name.clone())
                    .or_insert_with(|| ExprID::Expr(self.exprs.len())),
                name,
            ),
            ExprName::Infer => {
                let name = spec.expr.clone().meta().output_name()?.to_string();
                (
                    *self
                        .expr_name_to_id
                        .entry(name.clone())
                        .or_insert_with(|| ExprID::Expr(self.exprs.len())),
                    name,
                )
            }
            ExprName::None => (
                ExprID::Expr(self.exprs.len()),
                format!("UNNAMED_{}", self.exprs.len()),
            ),
        };
        self.exprs.push(DerivedCol {
            name: format!("__{}", col_name),
            width: spec.width,
            expr: spec.expr,
        });
        Ok(id)
    }

    pub fn register_scalar(&mut self, name: ExprName, x: Expr) -> LadduResult<ExprID> {
        self.register_expr(name, make_array(vec![x], 1))
    }
    pub fn register_cscalar(&mut self, name: ExprName, z: (Expr, Expr)) -> LadduResult<ExprID> {
        self.register_expr(name, make_array(vec![z.0, z.1], 2))
    }
    pub fn register_vector<const N: usize>(
        &mut self,
        name: ExprName,
        xs: [Expr; N],
    ) -> LadduResult<ExprID> {
        self.register_expr(name, make_array(xs.to_vec(), N))
    }
    pub fn register_cvector<const N: usize>(
        &mut self,
        name: ExprName,
        xs: [(Expr, Expr); N],
    ) -> LadduResult<ExprID> {
        self.register_expr(
            name,
            make_array(
                xs.into_iter().flat_map(|(re, im)| [re, im]).collect(),
                2 * N,
            ),
        )
    }
    pub fn register_matrix<const R: usize, const C: usize>(
        &mut self,
        name: ExprName,
        rows: [[Expr; C]; R],
    ) -> LadduResult<ExprID> {
        let mut flat = Vec::with_capacity(R * C);
        for c in 0..C {
            for r in 0..R {
                flat.push(rows[r][c].clone());
            }
        }
        self.register_expr(name, make_array(flat, R * C))
    }
    pub fn register_cmatrix<const R: usize, const C: usize>(
        &mut self,
        name: ExprName,
        rows: [[(Expr, Expr); C]; R],
    ) -> LadduResult<ExprID> {
        let mut flat = Vec::with_capacity(R * C);
        for c in 0..C {
            for r in 0..R {
                let (re, im) = &rows[r][c];
                flat.push(re.clone());
                flat.push(im.clone());
            }
        }
        self.register_expr(name, make_array(flat, 2 * R * C))
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use nalgebra::{Matrix2, Vector2};
    // use num::Complex;
    //
    // #[test]
    // fn test_parameters() {
    //     let parameters = vec![1.0, 2.0, 3.0];
    //     let constants = vec![4.0, 5.0, 6.0];
    //     let params = Parameters::new(&parameters, &constants);
    //
    //     assert_eq!(params.get(ParameterID::Parameter(0)), 1.0);
    //     assert_eq!(params.get(ParameterID::Parameter(1)), 2.0);
    //     assert_eq!(params.get(ParameterID::Parameter(2)), 3.0);
    //     assert_eq!(params.get(ParameterID::Constant(0)), 4.0);
    //     assert_eq!(params.get(ParameterID::Constant(1)), 5.0);
    //     assert_eq!(params.get(ParameterID::Constant(2)), 6.0);
    //     assert_eq!(params.len(), 3);
    // }
    //
    // #[test]
    // #[should_panic(expected = "Parameter has not been registered!")]
    // fn test_uninit_parameter() {
    //     let parameters = vec![1.0];
    //     let constants = vec![1.0];
    //     let params = Parameters::new(&parameters, &constants);
    //     params.get(ParameterID::Uninit);
    // }
    //
    // #[test]
    // fn test_resources_amplitude_management() {
    //     let mut resources = Resources::default();
    //
    //     let amp1 = resources.register_amplitude("amp1").unwrap();
    //     let amp2 = resources.register_amplitude("amp2").unwrap();
    //
    //     assert!(resources.active[amp1.1]);
    //     assert!(resources.active[amp2.1]);
    //
    //     resources.deactivate("amp1").unwrap();
    //     assert!(!resources.active[amp1.1]);
    //     assert!(resources.active[amp2.1]);
    //
    //     resources.activate("amp1").unwrap();
    //     assert!(resources.active[amp1.1]);
    //
    //     resources.deactivate_all();
    //     assert!(!resources.active[amp1.1]);
    //     assert!(!resources.active[amp2.1]);
    //
    //     resources.activate_all();
    //     assert!(resources.active[amp1.1]);
    //     assert!(resources.active[amp2.1]);
    //
    //     resources.isolate("amp1").unwrap();
    //     assert!(resources.active[amp1.1]);
    //     assert!(!resources.active[amp2.1]);
    // }
    //
    // #[test]
    // fn test_resources_parameter_registration() {
    //     let mut resources = Resources::default();
    //
    //     let param1 = resources.register_parameter(&ParameterLike::Parameter("param1".to_string()));
    //     let const1 = resources.register_parameter(&ParameterLike::Constant(1.0));
    //
    //     match param1 {
    //         ParameterID::Parameter(idx) => assert_eq!(idx, 0),
    //         _ => panic!("Expected Parameter variant"),
    //     }
    //
    //     match const1 {
    //         ParameterID::Constant(idx) => assert_eq!(idx, 0),
    //         _ => panic!("Expected Constant variant"),
    //     }
    // }
    //
    // #[test]
    // fn test_cache_scalar_operations() {
    //     let mut resources = Resources::default();
    //
    //     let scalar1 = resources.register_scalar(Some("test_scalar"));
    //     let scalar2 = resources.register_scalar(None);
    //     let scalar3 = resources.register_scalar(Some("test_scalar"));
    //
    //     resources.reserve_cache(1);
    //     let cache = &mut resources.caches[0];
    //
    //     cache.store_scalar(scalar1, 1.0);
    //     cache.store_scalar(scalar2, 2.0);
    //
    //     assert_eq!(cache.get_scalar(scalar1), 1.0);
    //     assert_eq!(cache.get_scalar(scalar2), 2.0);
    //     assert_eq!(cache.get_scalar(scalar3), 1.0);
    // }
    //
    // #[test]
    // fn test_cache_complex_operations() {
    //     let mut resources = Resources::default();
    //
    //     let complex1 = resources.register_complex_scalar(Some("test_complex"));
    //     let complex2 = resources.register_complex_scalar(None);
    //     let complex3 = resources.register_complex_scalar(Some("test_complex"));
    //
    //     resources.reserve_cache(1);
    //     let cache = &mut resources.caches[0];
    //
    //     let value1 = Complex::new(1.0, 2.0);
    //     let value2 = Complex::new(3.0, 4.0);
    //     cache.store_complex_scalar(complex1, value1);
    //     cache.store_complex_scalar(complex2, value2);
    //
    //     assert_eq!(cache.get_complex_scalar(complex1), value1);
    //     assert_eq!(cache.get_complex_scalar(complex2), value2);
    //     assert_eq!(cache.get_complex_scalar(complex3), value1);
    // }
    //
    // #[test]
    // fn test_cache_vector_operations() {
    //     let mut resources = Resources::default();
    //
    //     let vector_id1: VectorID<2> = resources.register_vector(Some("test_vector"));
    //     let vector_id2: VectorID<2> = resources.register_vector(None);
    //     let vector_id3: VectorID<2> = resources.register_vector(Some("test_vector"));
    //
    //     resources.reserve_cache(1);
    //     let cache = &mut resources.caches[0];
    //
    //     let value1 = Vector2::new(1.0, 2.0);
    //     let value2 = Vector2::new(3.0, 4.0);
    //     cache.store_vector(vector_id1, value1);
    //     cache.store_vector(vector_id2, value2);
    //
    //     assert_eq!(cache.get_vector(vector_id1), value1);
    //     assert_eq!(cache.get_vector(vector_id2), value2);
    //     assert_eq!(cache.get_vector(vector_id3), value1);
    // }
    //
    // #[test]
    // fn test_cache_complex_vector_operations() {
    //     let mut resources = Resources::default();
    //
    //     let complex_vector_id1: ComplexVectorID<2> =
    //         resources.register_complex_vector(Some("test_complex_vector"));
    //     let complex_vector_id2: ComplexVectorID<2> = resources.register_complex_vector(None);
    //     let complex_vector_id3: ComplexVectorID<2> =
    //         resources.register_complex_vector(Some("test_complex_vector"));
    //
    //     resources.reserve_cache(1);
    //     let cache = &mut resources.caches[0];
    //
    //     let value1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    //     let value2 = Vector2::new(Complex::new(5.0, 6.0), Complex::new(7.0, 8.0));
    //     cache.store_complex_vector(complex_vector_id1, value1);
    //     cache.store_complex_vector(complex_vector_id2, value2);
    //
    //     assert_eq!(cache.get_complex_vector(complex_vector_id1), value1);
    //     assert_eq!(cache.get_complex_vector(complex_vector_id2), value2);
    //     assert_eq!(cache.get_complex_vector(complex_vector_id3), value1);
    // }
    //
    // #[test]
    // fn test_cache_matrix_operations() {
    //     let mut resources = Resources::default();
    //
    //     let matrix_id1: MatrixID<2, 2> = resources.register_matrix(Some("test_matrix"));
    //     let matrix_id2: MatrixID<2, 2> = resources.register_matrix(None);
    //     let matrix_id3: MatrixID<2, 2> = resources.register_matrix(Some("test_matrix"));
    //
    //     resources.reserve_cache(1);
    //     let cache = &mut resources.caches[0];
    //
    //     let value1 = Matrix2::new(1.0, 2.0, 3.0, 4.0);
    //     let value2 = Matrix2::new(5.0, 6.0, 7.0, 8.0);
    //     cache.store_matrix(matrix_id1, value1);
    //     cache.store_matrix(matrix_id2, value2);
    //
    //     assert_eq!(cache.get_matrix(matrix_id1), value1);
    //     assert_eq!(cache.get_matrix(matrix_id2), value2);
    //     assert_eq!(cache.get_matrix(matrix_id3), value1);
    // }
    //
    // #[test]
    // fn test_cache_complex_matrix_operations() {
    //     let mut resources = Resources::default();
    //
    //     let complex_matrix_id1: ComplexMatrixID<2, 2> =
    //         resources.register_complex_matrix(Some("test_complex_matrix"));
    //     let complex_matrix_id2: ComplexMatrixID<2, 2> = resources.register_complex_matrix(None);
    //     let complex_matrix_id3: ComplexMatrixID<2, 2> =
    //         resources.register_complex_matrix(Some("test_complex_matrix"));
    //
    //     resources.reserve_cache(1);
    //     let cache = &mut resources.caches[0];
    //
    //     let value1 = Matrix2::new(
    //         Complex::new(1.0, 2.0),
    //         Complex::new(3.0, 4.0),
    //         Complex::new(5.0, 6.0),
    //         Complex::new(7.0, 8.0),
    //     );
    //     let value2 = Matrix2::new(
    //         Complex::new(9.0, 10.0),
    //         Complex::new(11.0, 12.0),
    //         Complex::new(13.0, 14.0),
    //         Complex::new(15.0, 16.0),
    //     );
    //     cache.store_complex_matrix(complex_matrix_id1, value1);
    //     cache.store_complex_matrix(complex_matrix_id2, value2);
    //
    //     assert_eq!(cache.get_complex_matrix(complex_matrix_id1), value1);
    //     assert_eq!(cache.get_complex_matrix(complex_matrix_id2), value2);
    //     assert_eq!(cache.get_complex_matrix(complex_matrix_id3), value1);
    // }
    //
    // #[test]
    // #[should_panic(expected = "Parameter was not initialized!")]
    // fn test_uninit_parameter_registration() {
    //     let mut resources = Resources::default();
    //     resources.register_parameter(&ParameterLike::Uninit);
    // }
    //
    // #[test]
    // fn test_duplicate_named_amplitude_registration_error() {
    //     let mut resources = Resources::default();
    //     assert!(resources.register_amplitude("test_amp").is_ok());
    //     assert!(resources.register_amplitude("test_amp").is_err());
    // }
}
