use std::sync::Arc;

use laddu_amplitudes::common::ComplexScalar;
#[cfg(feature = "mpi")]
use laddu_core::mpi::{finalize_mpi, get_world, use_mpi, LadduMPI};
use laddu_core::{
    data::{Dataset, DatasetMetadata, EventData},
    parameter,
    vectors::Vec4,
    Expression, LadduResult,
};
use laddu_extensions::{likelihoods::LikelihoodTerm, NLL};
use nalgebra::DVector;

struct CaseConfig {
    name: &'static str,
    expression: Expression,
    parameters: Vec<f64>,
    data_weights: Vec<f64>,
    mc_weights: Vec<f64>,
}

fn main() -> LadduResult<()> {
    #[cfg(feature = "mpi")]
    {
        use_mpi(true);
        if let Some(world) = get_world() {
            let _ = world;
            for case in case_configs() {
                let nll = make_test_nll(&case, None, None);
                let mpi_value = nll.evaluate_mpi_value(&case.parameters, &world)?;
                let mpi_gradient = nll.evaluate_mpi_gradient(&case.parameters, &world)?;
                if world.is_root() {
                    print_result_json("mpi", case.name, mpi_value, mpi_gradient.as_slice());
                }
            }
            finalize_mpi();
            return Ok(());
        }
        finalize_mpi();
    }

    for case in case_configs() {
        let nll = make_test_nll(&case, None, None);
        let local_value = nll.evaluate(&case.parameters)?;
        let local_gradient: DVector<f64> = nll.evaluate_gradient(&case.parameters)?;
        print_result_json("local", case.name, local_value, local_gradient.as_slice());
    }
    Ok(())
}

fn print_result_json(mode: &str, case: &str, value: f64, gradient: &[f64]) {
    let gradient_json = gradient
        .iter()
        .map(|g| format!("{g:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{{\"mode\":\"{mode}\",\"case\":\"{case}\",\"value\":{value:.17e},\"gradient\":[{gradient_json}]}}"
    );
}

fn make_test_nll(
    case: &CaseConfig,
    data_partition: Option<(usize, usize)>,
    mc_partition: Option<(usize, usize)>,
) -> Box<NLL> {
    let data_weights = match data_partition {
        Some((start, end)) => &case.data_weights[start..end],
        None => case.data_weights.as_slice(),
    };
    let mc_weights = match mc_partition {
        Some((start, end)) => &case.mc_weights[start..end],
        None => case.mc_weights.as_slice(),
    };

    let data = dataset_with_weights(data_weights);
    let mc = dataset_with_weights(mc_weights);

    NLL::new(&case.expression, &data, &mc, None).expect("NLL should construct")
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

fn case_configs() -> Vec<CaseConfig> {
    vec![
        CaseConfig {
            name: "product_norm",
            expression: {
                let amp_a = ComplexScalar::new("a", parameter!("a_re"), parameter!("a_im"))
                    .expect("ComplexScalar a should build");
                let amp_b = ComplexScalar::new("b", parameter!("b_re"), parameter!("b_im"))
                    .expect("ComplexScalar b should build");
                (&amp_a * &amp_b).norm_sqr()
            },
            parameters: vec![0.5, -0.25, 0.125, 0.4],
            data_weights: vec![1.0, 2.0, 3.0, 1.5, 0.75, 2.5, 1.25, 0.9],
            mc_weights: vec![0.5, 1.0, 1.5, 0.8, 1.2, 0.7, 1.1, 0.6],
        },
        CaseConfig {
            name: "sum_norm",
            expression: {
                let amp_a = ComplexScalar::new("a", parameter!("a_re"), parameter!("a_im"))
                    .expect("ComplexScalar a should build");
                let amp_b = ComplexScalar::new("b", parameter!("b_re"), parameter!("b_im"))
                    .expect("ComplexScalar b should build");
                (&amp_a + &amp_b).norm_sqr()
            },
            parameters: vec![-0.7, 0.3, 0.2, -0.1],
            data_weights: vec![2.4, 1.1, 0.9, 3.2, 1.8, 0.7, 2.9, 1.3],
            mc_weights: vec![1.4, 0.8, 2.1, 1.6, 0.9, 1.7, 1.2, 1.1],
        },
        CaseConfig {
            name: "sum_norm_with_fixed_b",
            expression: {
                let amp_a = ComplexScalar::new("a", parameter!("a_re"), parameter!("a_im"))
                    .expect("ComplexScalar a should build");
                let amp_b = ComplexScalar::new("b", parameter!("b_re"), parameter!("b_im"))
                    .expect("ComplexScalar b should build");
                let expression = (&amp_a + &amp_b).norm_sqr();
                expression
                    .fix_parameter("b_re", 0.15)
                    .expect("fix b_re should succeed");
                expression
                    .fix_parameter("b_im", -0.2)
                    .expect("fix b_im should succeed");
                expression
            },
            parameters: vec![0.8, -0.35],
            data_weights: vec![1.7, 1.2, 2.5, 0.6, 1.1, 3.4, 2.0, 0.8],
            mc_weights: vec![0.9, 1.3, 1.6, 0.7, 2.2, 1.0, 1.5, 0.95],
        },
    ]
}
