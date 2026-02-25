use std::sync::Arc;

use laddu_amplitudes::common::ComplexScalar;
use laddu_core::amplitudes::parameter;
use laddu_core::data::{Dataset, DatasetMetadata, EventData};
use laddu_core::utils::vectors::Vec4;
use laddu_core::LadduResult;
use laddu_extensions::likelihoods::LikelihoodTerm;
use laddu_extensions::NLL;
use nalgebra::DVector;

#[cfg(feature = "mpi")]
use laddu_core::mpi::{finalize_mpi, get_world, use_mpi, LadduMPI};

fn main() -> LadduResult<()> {
    #[cfg(feature = "mpi")]
    {
        use_mpi(true);
        if let Some(world) = get_world() {
            let _ = world;
            let (nll, params) = make_test_nll(None, None);
            let mpi_value = nll.evaluate_mpi_value(&params, &world);
            let mpi_gradient = nll.evaluate_mpi_gradient(&params, &world);
            if world.is_root() {
                print_result_json("mpi", mpi_value, mpi_gradient.as_slice());
            }
            finalize_mpi();
            return Ok(());
        }
        finalize_mpi();
    }

    let (nll, params) = make_test_nll(None, None);
    let local_value = nll.evaluate(&params)?;
    let local_gradient: DVector<f64> = nll.evaluate_gradient(&params)?;
    print_result_json("local", local_value, local_gradient.as_slice());
    Ok(())
}

fn print_result_json(mode: &str, value: f64, gradient: &[f64]) {
    let gradient_json = gradient
        .iter()
        .map(|g| format!("{g:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("{{\"mode\":\"{mode}\",\"value\":{value:.17e},\"gradient\":[{gradient_json}]}}");
}

fn make_test_nll(
    data_partition: Option<(&[f64], (usize, usize))>,
    mc_partition: Option<(&[f64], (usize, usize))>,
) -> (Box<NLL>, Vec<f64>) {
    let amp_a = ComplexScalar::new("a", parameter("a_re"), parameter("a_im"))
        .expect("ComplexScalar a should build");
    let amp_b = ComplexScalar::new("b", parameter("b_re"), parameter("b_im"))
        .expect("ComplexScalar b should build");
    let expr = (&amp_a * &amp_b).norm_sqr();

    let data = match data_partition {
        Some((weights, (start, end))) => dataset_with_weights(&weights[start..end]),
        None => dataset_with_weights(&[1.0, 2.0, 3.0, 1.5, 0.75, 2.5, 1.25, 0.9]),
    };
    let mc = match mc_partition {
        Some((weights, (start, end))) => dataset_with_weights(&weights[start..end]),
        None => dataset_with_weights(&[0.5, 1.0, 1.5, 0.8, 1.2, 0.7, 1.1, 0.6]),
    };

    let nll = NLL::new(&expr, &data, &mc).expect("NLL should construct");
    (nll, vec![0.5, -0.25, 0.125, 0.4])
}

fn dataset_with_weights(weights: &[f64]) -> Arc<Dataset> {
    let metadata = Arc::new(DatasetMetadata::default());
    let events = weights
        .iter()
        .map(|&weight| {
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 1.0)],
                aux: vec![],
                weight,
            })
        })
        .collect();
    Arc::new(Dataset::new_with_metadata(events, metadata))
}
