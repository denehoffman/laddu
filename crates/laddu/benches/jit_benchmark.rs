//! Benchmarks interpreted and JIT-compiled scalar evaluation.
#![allow(
    missing_docs,
    reason = "criterion generates an undocumented public function"
)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use laddu::{
    Expr, Parameter,
    compile::CompiledModel,
    runtime::{CpuBackend, CpuExecutionMode},
};

fn arithmetic_model() -> CompiledModel {
    let mut expression = Expr::from(Parameter::free("p0").with_initial(0.25));
    for index in 1..32 {
        let parameter =
            Expr::from(Parameter::free(format!("p{index}")).with_initial(index as f64 / 32.0));
        expression = expression * 1.000_001 + parameter;
    }
    CompiledModel::from_expr(&expression).unwrap()
}

fn scalar_jit_benchmark(criterion: &mut Criterion) {
    let model = arithmetic_model();
    let params = model.params().default_values();
    let automatic = CpuBackend.prepare(&model);
    let interpreted = CpuBackend.prepare_with_execution_mode(&model, CpuExecutionMode::Interpreter);
    assert_eq!(
        automatic.evaluate(&params).unwrap(),
        interpreted.evaluate(&params).unwrap()
    );

    let mut group = criterion.benchmark_group("Scalar Kernel Execution");
    for (name, plan) in [("JIT", automatic), ("Interpreter", interpreted)] {
        group.bench_with_input(BenchmarkId::new("Backend", name), &plan, |bencher, plan| {
            bencher.iter(|| black_box(plan.evaluate(black_box(&params)).unwrap()))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(500);
    targets = scalar_jit_benchmark
}
criterion_main!(benches);
