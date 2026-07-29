//! Benchmarks for weighted and unweighted channel event generation.
#![allow(
    missing_docs,
    reason = "criterion generates an undocumented public function"
)]

use criterion::{Criterion, criterion_group, criterion_main};
use laddu_compile::CompiledModel;
use laddu_expr::Expr;
use laddu_generation::{
    ChannelGenerator, EnvelopeMode, ModelEvaluator, UnweightedConfig, WeightedConfig,
};
use laddu_physics::{channel::Channel, quantum::ParticleProperties, vectors::RealVec4};
use laddu_runtime::Execution;

fn generator() -> ChannelGenerator {
    let mut channel = Channel::new("generation benchmark");
    channel
        .edge("parent")
        .properties(&ParticleProperties::unknown().with_mass(2.0))
        .initial_p4(RealVec4::new(2.0, 0.0, 0.0, 0.0));
    channel
        .edge("a")
        .properties(&ParticleProperties::unknown().with_mass(0.2))
        .output();
    channel
        .edge("b")
        .properties(&ParticleProperties::unknown().with_mass(0.4))
        .output();
    channel
        .vertex("decay")
        .incoming(["parent"])
        .outgoing(["a", "b"]);
    ChannelGenerator::new(channel).unwrap()
}

fn evaluator() -> ModelEvaluator {
    let model = CompiledModel::from_expr(&Expr::from(1.0)).unwrap();
    ModelEvaluator::prepare(
        &model,
        model.params().default_values(),
        &Execution::default(),
    )
    .unwrap()
}

fn benchmarks(c: &mut Criterion) {
    let generator = generator();
    let evaluator = evaluator();

    c.bench_function("generation/weighted_16384", |b| {
        b.iter(|| {
            generator
                .generate_weighted_dataset(
                    WeightedConfig {
                        events: 16_384,
                        memory: laddu_runtime::MemoryBudget::Bytes(2 << 20),
                        seed: 17,
                        diagnostics: false,
                    },
                    Some(&evaluator),
                )
                .unwrap()
        })
    });

    c.bench_function("generation/unweighted_strict_4096", |b| {
        b.iter(|| {
            generator
                .generate_unweighted_dataset(
                    UnweightedConfig {
                        envelope: EnvelopeMode::Strict { max_weight: 0.1 },
                        ..UnweightedConfig::new(4_096).with_max_proposals(100_000)
                    },
                    &evaluator,
                )
                .unwrap()
        })
    });
}

criterion_group!(generation, benchmarks);
criterion_main!(generation);
