use std::sync::Arc;

use crate::{
    LadduDataError, LadduDataResult,
    columns::ColumnValue,
    data::event::{BatchAssembler, Event, EventBatch},
};

#[derive(Clone)]
pub(super) enum DatasetOp {
    Filter(Arc<dyn Fn(Event<'_>) -> bool + Send + Sync>),
    Subsample {
        fraction: f64,
        seed: u64,
    },
    Bootstrap {
        seed: u64,
        grouping: Option<Arc<BootstrapGrouping>>,
    },
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
        let event_id = batch.row_ids().map_or(base + row as u64, |ids| ids[row]);
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
                DatasetOp::Bootstrap { seed, grouping } => {
                    let key = match grouping {
                        Some(groups) => groups.key(batch, row)?,
                        None => event_id,
                    };
                    let k = poisson1_from_hash(*seed, key);
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

    let store_weights = batch.weights_column().is_some()
        || ops
            .iter()
            .any(|op| matches!(op, DatasetOp::Bootstrap { .. }));

    let mut assembler =
        BatchAssembler::with_weight_mode(Arc::clone(batch.schema()), batch.len(), store_weights);

    eval_batch(batch, ops, base, |ev| {
        assembler.push_borrowed(ev, store_weights)
    })?;

    assembler.finish()
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

/// Stable exact-integer group keys. The hash encoding is versioned by this API.
#[derive(Clone)]
pub(super) struct BootstrapGrouping {
    pub(super) columns: Arc<[String]>,
    pub(super) namespace: String,
}

impl BootstrapGrouping {
    fn key(&self, batch: &EventBatch, row: usize) -> LadduDataResult<u64> {
        // FNV-1a over length-framed UTF-8 names and canonical signed 128-bit integers.
        // Integer widths do not change a logical key; uint64 never passes through f64.
        let mut hash = 0xcbf29ce484222325_u64;
        let mut update = |bytes: &[u8]| {
            for byte in bytes {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
            }
        };
        update(b"laddu-bootstrap-groups-v1");
        update(&(self.namespace.len() as u64).to_le_bytes());
        update(self.namespace.as_bytes());
        for name in self.columns.iter() {
            update(&(name.len() as u64).to_le_bytes());
            update(name.as_bytes());
            let column = batch
                .column_named(name)
                .ok_or_else(|| LadduDataError::MissingColumn(crate::Name::from(name.as_str())))?;
            let value: i128 = match column.at(row) {
                ColumnValue::I8(value) => value.into(),
                ColumnValue::U8(value) => value.into(),
                ColumnValue::I16(value) => value.into(),
                ColumnValue::U16(value) => value.into(),
                ColumnValue::I32(value) => value.into(),
                ColumnValue::U32(value) => value.into(),
                ColumnValue::I64(value) => value.into(),
                ColumnValue::U64(value) => value.into(),
            };
            update(&value.to_le_bytes());
        }
        Ok(hash)
    }
}
