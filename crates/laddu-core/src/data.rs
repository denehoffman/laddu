use auto_ops::impl_op_ex;
use fastrand::Rng;
use polars::prelude::*;
use std::sync::Arc;

#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};

#[cfg(feature = "mpi")]
use crate::mpi::LadduMPI;

use crate::utils::get_bin_edges;
use crate::LadduResult;

pub fn test_dataset() -> LadduResult<Dataset> {
    Ok(Dataset::new(
        df!(
            "beam_e" => [8.747],
            "beam_px" => [0.0],
            "beam_py" => [0.0],
            "beam_pz" => [8.747],
            "proton_e" => [1.10334],
            "proton_px" => [0.119],
            "proton_py" => [0.374],
            "proton_pz" => [0.222],
            "kshort1_e" => [3.13671],
            "kshort1_px" => [-0.112],
            "kshort1_py" => [0.293],
            "kshort1_pz" => [3.081],
            "kshort2_e" => [5.50925],
            "kshort2_px" => [-0.007],
            "kshort2_py" => [-0.667],
            "kshort2_pz" => [5.446],
            "weight" => [0.48],
            "pol_angle" => [0.0570808],
            "pol_magnitude" => [0.385628],
        )?
        .lazy(),
    ))
}

#[inline]
fn count_rows(lf: &LazyFrame) -> LadduResult<usize> {
    let df = lf.clone().select([len().alias("__n")]).collect()?;
    Ok(df.column("__n")?.u64()?.get(0).unwrap_or(0) as usize)
}

/// A collection of [`Event`]s.
#[derive(Clone)]
pub struct Dataset {
    lf: LazyFrame,
}

impl Dataset {
    pub fn new(lazyframe: LazyFrame) -> Self {
        Self { lf: lazyframe }
    }

    // TODO: write version with rest-frame boost
    pub fn open<T: AsRef<str>>(path: T) -> LadduResult<Self> {
        Ok(Self::new(LazyFrame::scan_parquet(
            PlPath::new(path.as_ref()),
            Default::default(),
        )?))
    }

    /// The number of [`Event`]s in the [`Dataset`].
    pub fn n_events(&self) -> LadduResult<usize> {
        count_rows(&self.lf)
    }

    /// Extract a list of weights over each [`Event`] in the [`Dataset`].
    pub fn weights(&self) -> LadduResult<Vec<f64>> {
        Ok(self
            .lf
            .clone()
            .select([col("weight")])
            .collect()?
            .column("weight")?
            .f64()?
            .to_vec_null_aware()
            .left()
            .unwrap())
    }

    /// Returns the sum of the weights for each [`Event`] in the [`Dataset`].
    pub fn n_events_weighted(&self) -> LadduResult<f64> {
        Ok(self
            .lf
            .clone()
            .select([col("weight").sum().alias("__n_weighted")])
            .collect()?
            .column("__n_weighted")?
            .f64()?
            .get(0)
            .unwrap_or(0.0))
    }

    /// Generate a new dataset with the same length by resampling the events in the original datset
    /// with replacement. This can be used to perform error analysis via the bootstrap method.
    pub fn bootstrap(&self, seed: usize) -> LadduResult<Dataset> {
        let n = self.n_events()?;
        if n == 0 {
            return Ok(self.clone());
        }
        let (start, count) = crate::mpi::get_range_for_rank(n);
        let mut rng = Rng::with_seed(seed as u64);
        let idx: Vec<u64> = (0..n).map(|_| rng.u64(..(n as u64))).collect();
        let idx_for_rank: Vec<u64> = idx[start..(start + count)].to_vec();
        let idx_df = df!("__row" => idx_for_rank)?;
        let base = self.lf.clone().with_row_index("__row", None);
        Ok(Dataset::new(
            idx_df
                .lazy()
                .join(
                    base,
                    [col("__row")],
                    [col("__row")],
                    JoinArgs::new(JoinType::Left),
                )
                .drop(cols(["__row"])),
        ))
    }

    /// Filter the [`Dataset`] by a given [`VariableExpression`], selecting events for which
    /// the expression returns `true`.
    pub fn filter(&self, expression: &Expr) -> LadduResult<Dataset> {
        Ok(Dataset::new(self.lf.clone().filter(expression.clone())))
    }

    /// Bin a [`Dataset`] by the value of the given [`Variable`] into a number of `bins` within the
    /// given `range`.
    pub fn bin_by<V>(
        &self,
        expression: &Expr,
        bins: usize,
        limits: (f64, f64),
    ) -> LadduResult<(Vec<Dataset>, Vec<f64>)> {
        let bin_edges = get_bin_edges(bins, limits);
        Ok((
            bin_edges
                .windows(2)
                .map(|window| {
                    Dataset::new(
                        self.lf.clone().filter(
                            expression
                                .clone()
                                .gt_eq(lit(window[0]))
                                .logical_and(expression.clone().lt(lit(window[1]))),
                        ),
                    )
                })
                .collect::<Vec<Dataset>>(),
            bin_edges,
        ))
    }

    /// Boost all the four-momenta in all [`Event`]s to the rest frame of the given set of
    /// four-momenta by indices.
    pub fn boost_to_rest_frame_of<T: AsRef<[usize]> + Sync>(&self, _indices: T) -> Arc<Dataset> {
        todo!()
    }
    /// Evaluate a [`Variable`] on every event in the [`Dataset`].
    pub fn evaluate(&self, variable: &Expr) -> LadduResult<Vec<f64>> {
        Ok(self
            .lf
            .clone()
            .select([variable.clone()])
            .collect()?
            .column(&variable.clone().meta().output_name()?)?
            .f64()?
            .to_vec_null_aware()
            .left()
            .unwrap_or_else(Vec::default))
    }
}

impl_op_ex!(+ |a: &Dataset, b: &Dataset| ->  LadduResult<Dataset> { Ok(Dataset::new(concat([a.lf.clone(), b.lf.clone()], Default::default())?))});

#[cfg(test)]
mod tests {
    // use crate::Mass;

    use super::*;
    // use approx::{assert_relative_eq, assert_relative_ne};
    // use serde::{Deserialize, Serialize};

    #[test]
    fn test_dataset_size_check() {
        let mut dataset = test_dataset().unwrap();
        assert_eq!(dataset.n_events().unwrap(), 0);
        dataset = (dataset + test_dataset().unwrap()).unwrap();
        assert_eq!(dataset.n_events().unwrap(), 1);
    }

    //
    // #[test]
    // fn test_dataset_weights() {
    //     let mut dataset = Dataset::default();
    //     dataset.events.push(Arc::new(test_event()));
    //     dataset.events.push(Arc::new(Event {
    //         p4s: test_event().p4s,
    //         aux: test_event().aux,
    //         weight: 0.52,
    //     }));
    //     let weights = dataset.weights();
    //     assert_eq!(weights.len(), 2);
    //     assert_relative_eq!(weights[0], 0.48);
    //     assert_relative_eq!(weights[1], 0.52);
    //     assert_relative_eq!(dataset.n_events_weighted(), 1.0);
    // }
    //
    // #[test]
    // fn test_dataset_filtering() {
    //     let mut dataset = Dataset::default();
    //     dataset.events.push(Arc::new(Event {
    //         p4s: vec![Vec3::new(0.0, 0.0, 5.0).with_mass(0.0)],
    //         aux: vec![],
    //         weight: 1.0,
    //     }));
    //     dataset.events.push(Arc::new(Event {
    //         p4s: vec![Vec3::new(0.0, 0.0, 5.0).with_mass(0.5)],
    //         aux: vec![],
    //         weight: 1.0,
    //     }));
    //     dataset.events.push(Arc::new(Event {
    //         p4s: vec![Vec3::new(0.0, 0.0, 5.0).with_mass(1.1)],
    //         // HACK: using 1.0 messes with this test because the eventual computation gives a mass
    //         // slightly less than 1.0
    //         aux: vec![],
    //         weight: 1.0,
    //     }));
    //
    //     let mass = Mass::new([0]);
    //     let expression = mass.gt(0.0).and(&mass.lt(1.0));
    //
    //     let filtered = dataset.filter(&expression);
    //     assert_eq!(filtered.n_events(), 1);
    //     assert_relative_eq!(mass.value(&filtered[0]), 0.5, epsilon = f64::EPSILON.sqrt());
    // }
    //
    // #[test]
    // fn test_dataset_boost() {
    //     let dataset = test_dataset();
    //     let dataset_boosted = dataset.boost_to_rest_frame_of([1, 2, 3]);
    //     let p4_sum = dataset_boosted[0].get_p4_sum([1, 2, 3]);
    //     assert_relative_eq!(p4_sum.px(), 0.0, epsilon = f64::EPSILON.sqrt());
    //     assert_relative_eq!(p4_sum.py(), 0.0, epsilon = f64::EPSILON.sqrt());
    //     assert_relative_eq!(p4_sum.pz(), 0.0, epsilon = f64::EPSILON.sqrt());
    // }
    //
    // #[test]
    // fn test_dataset_evaluate() {
    //     let dataset = test_dataset();
    //     let mass = Mass::new([1]);
    //     assert_relative_eq!(dataset.evaluate(&mass)[0], 1.007);
    // }
    //
    // #[test]
    // fn test_binned_dataset() {
    //     let dataset = Dataset::new(vec![
    //         Arc::new(Event {
    //             p4s: vec![Vec3::new(0.0, 0.0, 1.0).with_mass(1.0)],
    //             aux: vec![],
    //             weight: 1.0,
    //         }),
    //         Arc::new(Event {
    //             p4s: vec![Vec3::new(0.0, 0.0, 2.0).with_mass(2.0)],
    //             aux: vec![],
    //             weight: 2.0,
    //         }),
    //     ]);
    //
    //     #[derive(Clone, Serialize, Deserialize, Debug)]
    //     struct BeamEnergy;
    //     impl Display for BeamEnergy {
    //         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    //             write!(f, "BeamEnergy")
    //         }
    //     }
    //     #[typetag::serde]
    //     impl Variable for BeamEnergy {
    //         fn value(&self, event: &Event) -> f64 {
    //             event.p4s[0].e()
    //         }
    //     }
    //     assert_eq!(BeamEnergy.to_string(), "BeamEnergy");
    //
    //     // Test binning by first particle energy
    //     let binned = dataset.bin_by(BeamEnergy, 2, (0.0, 3.0));
    //
    //     assert_eq!(binned.n_bins(), 2);
    //     assert_eq!(binned.edges().len(), 3);
    //     assert_relative_eq!(binned.edges()[0], 0.0);
    //     assert_relative_eq!(binned.edges()[2], 3.0);
    //     assert_eq!(binned[0].n_events(), 1);
    //     assert_relative_eq!(binned[0].n_events_weighted(), 1.0);
    //     assert_eq!(binned[1].n_events(), 1);
    //     assert_relative_eq!(binned[1].n_events_weighted(), 2.0);
    // }
    //
    // #[test]
    // fn test_dataset_bootstrap() {
    //     let mut dataset = test_dataset();
    //     dataset.events.push(Arc::new(Event {
    //         p4s: test_event().p4s.clone(),
    //         aux: test_event().aux.clone(),
    //         weight: 1.0,
    //     }));
    //     assert_relative_ne!(dataset[0].weight, dataset[1].weight);
    //
    //     let bootstrapped = dataset.bootstrap(43);
    //     assert_eq!(bootstrapped.n_events(), dataset.n_events());
    //     assert_relative_eq!(bootstrapped[0].weight, bootstrapped[1].weight);
    //
    //     // Test empty dataset bootstrap
    //     let empty_dataset = Dataset::default();
    //     let empty_bootstrap = empty_dataset.bootstrap(43);
    //     assert_eq!(empty_bootstrap.n_events(), 0);
    // }
    // #[test]
    // fn test_event_display() {
    //     let event = test_event();
    //     let display_string = format!("{}", event);
    //     assert_eq!(
    //         display_string,
    //         "Event:\n  p4s:\n    [e = 8.74700; p = (0.00000, 0.00000, 8.74700); m = 0.00000]\n    [e = 1.10334; p = (0.11900, 0.37400, 0.22200); m = 1.00700]\n    [e = 3.13671; p = (-0.11200, 0.29300, 3.08100); m = 0.49800]\n    [e = 5.50925; p = (-0.00700, -0.66700, 5.44600); m = 0.49800]\n  eps:\n    [0.385, 0.022, 0]\n  weight:\n    0.48\n"
    //     );
    // }
}
