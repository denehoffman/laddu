use std::sync::Arc;

use laddu_physics::vectors::RealVec4;

use crate::{
    LadduDataResult,
    data::event::{Event, EventBatch},
};

#[derive(Clone)]
pub(super) enum DatasetOp {
    Filter(Arc<dyn Fn(Event<'_>) -> bool + Send + Sync>),
    Subsample { fraction: f64, seed: u64 },
    Bootstrap { seed: u64 },
}

pub(super) fn eval_batch<F>(
    batch: &EventBatch,
    ops: &[DatasetOp],
    base: u64,
    mut f: F,
) -> LadduDataResult<()>
where
    F: FnMut(Event<'_>) -> LadduDataResult<()>,
{
    'rows: for row in 0..batch.len() {
        let event_id = base + row as u64;
        let mut weight = batch.weights_at(row);

        for op in ops {
            match op {
                DatasetOp::Filter(pred) => {
                    let ev = Event { batch, row, weight };
                    if !pred(ev) {
                        continue 'rows;
                    }
                }
                DatasetOp::Subsample { fraction, seed } => {
                    if uniform_hash_01(*seed, event_id) >= *fraction {
                        continue 'rows;
                    }
                }
                DatasetOp::Bootstrap { seed } => {
                    let k = poisson1_from_hash(*seed, event_id);
                    weight *= k as f64;
                }
            }
        }

        f(Event { batch, row, weight })?;
    }
    Ok(())
}

pub(super) fn materialize_batch(
    batch: &EventBatch,
    ops: &[DatasetOp],
    base: u64,
) -> LadduDataResult<EventBatch> {
    if ops.is_empty() {
        return Ok(batch.clone());
    }

    let schema = Arc::clone(batch.schema());

    let store_weights = batch.weights_column().is_some()
        || ops
            .iter()
            .any(|op| matches!(op, DatasetOp::Bootstrap { .. }));

    let mut p4s: Vec<Vec<RealVec4>> = (0..schema.n_p4s())
        .map(|_| Vec::with_capacity(batch.len()))
        .collect();
    let mut scalars: Vec<Vec<f64>> = (0..schema.n_scalars())
        .map(|_| Vec::with_capacity(batch.len()))
        .collect();
    let mut weights = if store_weights {
        Some(Vec::with_capacity(batch.len()))
    } else {
        None
    };

    eval_batch(batch, ops, base, |ev| {
        for (col, p4) in p4s.iter_mut().enumerate() {
            p4.push(ev.p4(col));
        }

        for (col, scalar) in scalars.iter_mut().enumerate() {
            scalar.push(ev.scalar(col));
        }

        if let Some(weights) = weights.as_mut() {
            weights.push(ev.weight());
        }

        Ok(())
    })?;

    let p4s = p4s.into_iter().map(Arc::from).collect();
    let scalars = scalars.into_iter().map(Arc::from).collect();
    let weights = weights.map(Arc::from);

    EventBatch::new(schema, p4s, scalars, weights)
}

pub(super) fn uniform_hash_01(seed: u64, index: u64) -> f64 {
    let x = splitmix64(seed ^ index);
    ((x >> 11) as f64) / ((1_u64 << 53) as f64)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);

    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub(super) fn poisson1_from_hash(seed: u64, index: u64) -> u32 {
    let u = uniform_hash_01(seed, index);

    let mut k = 0;
    let mut p = (-1.0_f64).exp();
    let mut cdf = p;

    while u > cdf {
        k += 1;
        p /= k as f64;
        cdf += p;
    }

    k
}
