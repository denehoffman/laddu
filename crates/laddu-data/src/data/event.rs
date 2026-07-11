use std::sync::Arc;

use laddu_physics::vectors::RealVec4;

use crate::{LadduDataError, LadduDataResult, schema::Schema};

#[derive(Clone, Debug)]
pub struct EventBatch {
    schema: Arc<Schema>,
    len: usize,
    p4s: Arc<[Arc<[RealVec4]>]>,
    scalars: Arc<[Arc<[f64]>]>,
    weights: Option<Arc<[f64]>>,
}

impl EventBatch {
    pub fn new(
        schema: Arc<Schema>,
        p4s: Vec<Arc<[RealVec4]>>,
        scalars: Vec<Arc<[f64]>>,
        weights: Option<Arc<[f64]>>,
    ) -> LadduDataResult<Self> {
        if p4s.len() != schema.n_p4s() {
            return Err(LadduDataError::Schema(
                "wrong number of vec4 columns".into(),
            ));
        }

        if scalars.len() != schema.n_scalars() {
            return Err(LadduDataError::Schema(
                "wrong number of scalar columns".into(),
            ));
        }

        let len = infer_len(&p4s, &scalars, weights.as_deref())?;

        Ok(Self {
            schema,
            len,
            p4s: p4s.into(),
            scalars: scalars.into(),
            weights,
        })
    }

    pub fn from_events<I>(schema: Arc<Schema>, events: I) -> LadduDataResult<Self>
    where
        I: IntoIterator<Item = OwnedEvent>,
    {
        let mut builder = EventBatchBuilder::new(schema);
        builder.extend(events)?;
        builder.finish()
    }

    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn vec4_column(&self, index: usize) -> &[RealVec4] {
        &self.p4s[index]
    }

    pub fn scalar_column(&self, index: usize) -> &[f64] {
        &self.scalars[index]
    }

    pub fn weights_column(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    pub fn vec4_column_named(&self, name: &str) -> Option<&[RealVec4]> {
        let i = self.schema.p4_index(name)?;
        Some(self.vec4_column(i))
    }

    pub fn scalar_column_named(&self, name: &str) -> Option<&[f64]> {
        let i = self.schema.scalar_index(name)?;
        Some(self.scalar_column(i))
    }

    pub fn p4_at(&self, col: usize, row: usize) -> RealVec4 {
        self.p4s[col][row]
    }

    pub fn scalar_at(&self, col: usize, row: usize) -> f64 {
        self.scalars[col][row]
    }

    pub fn weights_at(&self, row: usize) -> f64 {
        self.weights.as_ref().map_or(1.0, |w| w[row])
    }

    pub fn event(&self, row: usize) -> BatchEvent<'_> {
        BatchEvent { batch: self, row }
    }

    pub fn select(&self, rows: &[usize]) -> Self {
        let p4s = self
            .p4s
            .iter()
            .map(|col| rows.iter().map(|&i| col[i]).collect())
            .collect();

        let scalars = self
            .scalars
            .iter()
            .map(|col| rows.iter().map(|&i| col[i]).collect())
            .collect();

        let weights = self
            .weights
            .as_ref()
            .map(|w| rows.iter().map(|&i| w[i]).collect());

        Self {
            schema: Arc::clone(&self.schema),
            len: rows.len(),
            p4s,
            scalars,
            weights,
        }
    }

    pub fn filter<F>(&self, keep: F) -> Self
    where
        F: Fn(BatchEvent<'_>) -> bool,
    {
        let rows: Vec<usize> = (0..self.len).filter(|&i| keep(self.event(i))).collect();

        self.select(&rows)
    }

    pub fn reweight<F>(&self, f: F) -> Self
    where
        F: Fn(usize, f64) -> f64,
    {
        let weights: Arc<[f64]> = (0..self.len).map(|i| f(i, self.weights_at(i))).collect();

        Self {
            schema: Arc::clone(&self.schema),
            len: self.len,
            p4s: Arc::clone(&self.p4s),
            scalars: Arc::clone(&self.scalars),
            weights: Some(weights),
        }
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start <= end);
        assert!(end <= self.len);

        if start == 0 && end == self.len {
            return self.clone();
        }

        let p4s = self
            .p4s
            .iter()
            .map(|col| Arc::<[RealVec4]>::from(&col[start..end]))
            .collect();

        let scalars = self
            .scalars
            .iter()
            .map(|col| Arc::<[f64]>::from(&col[start..end]))
            .collect();

        let weights = self
            .weights
            .as_ref()
            .map(|w| Arc::<[f64]>::from(&w[start..end]));

        Self {
            schema: Arc::clone(&self.schema),
            len: end - start,
            p4s,
            scalars,
            weights,
        }
    }

    pub fn concat(batches: &[Self]) -> LadduDataResult<Self> {
        if batches.is_empty() {
            return Err(LadduDataError::InvalidArgument(
                "cannot concatenate zero batches",
            ));
        }

        let schema = Arc::clone(&batches[0].schema);
        let len: usize = batches.iter().map(|b| b.len).sum();

        for batch in batches {
            if schema != batch.schema {
                return Err(LadduDataError::Schema(
                    "cannot concatenate batches with different schemas".into(),
                ));
            }
        }

        let mut p4s = Vec::with_capacity(schema.n_p4s());

        for col in 0..schema.n_p4s() {
            let mut out = Vec::with_capacity(len);
            for batch in batches {
                out.extend_from_slice(batch.vec4_column(col));
            }
            p4s.push(Arc::from(out));
        }

        let mut scalars = Vec::with_capacity(schema.n_scalars());

        for col in 0..schema.n_scalars() {
            let mut out = Vec::with_capacity(len);
            for batch in batches {
                out.extend_from_slice(batch.scalar_column(col));
            }
            scalars.push(Arc::from(out));
        }

        let any_weights = batches.iter().any(|b| b.weights.is_some());

        let weights = if any_weights {
            let mut out = Vec::with_capacity(len);
            for batch in batches {
                for i in 0..batch.len {
                    out.push(batch.weights_at(i));
                }
            }
            Some(Arc::from(out))
        } else {
            None
        };

        Ok(Self {
            schema,
            len,
            p4s: p4s.into(),
            scalars: scalars.into(),
            weights,
        })
    }
}

fn infer_len(
    vec4s: &[Arc<[RealVec4]>],
    scalars: &[Arc<[f64]>],
    weight: Option<&[f64]>,
) -> LadduDataResult<usize> {
    let len = vec4s
        .first()
        .map(|c| c.len())
        .or_else(|| scalars.first().map(|c| c.len()))
        .or_else(|| weight.map(|w| w.len()))
        .unwrap_or(0);

    for col in vec4s {
        if col.len() != len {
            return Err(LadduDataError::Schema(
                "inconsistent vec4 column length".into(),
            ));
        }
    }

    for col in scalars {
        if col.len() != len {
            return Err(LadduDataError::Schema(
                "inconsistent scalar column length".into(),
            ));
        }
    }

    if let Some(w) = weight
        && w.len() != len {
            return Err(LadduDataError::Schema("inconsistent weight length".into()));
        }

    Ok(len)
}

#[derive(Copy, Clone, Debug)]
pub struct BatchEvent<'a> {
    batch: &'a EventBatch,
    row: usize,
}

impl<'a> BatchEvent<'a> {
    pub fn row(&self) -> usize {
        self.row
    }

    pub fn batch(&self) -> &'a EventBatch {
        self.batch
    }

    pub fn p4(&self, col: usize) -> RealVec4 {
        self.batch.p4_at(col, self.row)
    }

    pub fn scalar(&self, col: usize) -> f64 {
        self.batch.scalar_at(col, self.row)
    }

    pub fn weight(&self) -> f64 {
        self.batch.weights_at(self.row)
    }

    pub fn p4_named(&self, name: &str) -> Option<RealVec4> {
        let col = self.batch.schema.p4_index(name)?;
        Some(self.p4(col))
    }

    pub fn scalar_named(&self, name: &str) -> Option<f64> {
        let col = self.batch.schema.scalar_index(name)?;
        Some(self.scalar(col))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Event<'a> {
    pub(super) batch: &'a EventBatch,
    pub(super) row: usize,
    pub(super) weight: f64,
}

impl<'a> Event<'a> {
    pub fn row(&self) -> usize {
        self.row
    }

    pub fn p4(&self, col: usize) -> RealVec4 {
        self.batch.p4_at(col, self.row)
    }

    pub fn scalar(&self, col: usize) -> f64 {
        self.batch.scalar_at(col, self.row)
    }

    pub fn weight(&self) -> f64 {
        self.weight
    }

    pub fn p4_named(&self, name: &str) -> Option<RealVec4> {
        let col = self.batch.schema.p4_index(name)?;
        Some(self.p4(col))
    }

    pub fn scalar_named(&self, name: &str) -> Option<f64> {
        let col = self.batch.schema.scalar_index(name)?;
        Some(self.scalar(col))
    }
}

#[derive(Clone, Debug)]
pub struct OwnedEvent {
    pub p4s: Vec<RealVec4>,
    pub scalars: Vec<f64>,
    pub weight: Option<f64>,
}

impl OwnedEvent {
    pub fn new(p4s: Vec<RealVec4>, scalars: Vec<f64>) -> Self {
        Self {
            p4s,
            scalars,
            weight: None,
        }
    }

    pub fn weighted(p4s: Vec<RealVec4>, scalars: Vec<f64>, weight: f64) -> Self {
        Self {
            p4s,
            scalars,
            weight: Some(weight),
        }
    }
}

pub struct EventBatchBuilder {
    schema: Arc<Schema>,
    p4s: Vec<Vec<RealVec4>>,
    scalars: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
    len: usize,
}

impl EventBatchBuilder {
    pub fn new(schema: Arc<Schema>) -> Self {
        let p4s = (0..schema.n_p4s()).map(|_| Vec::new()).collect();
        let scalars = (0..schema.n_scalars()).map(|_| Vec::new()).collect();

        Self {
            schema,
            p4s,
            scalars,
            weights: None,
            len: 0,
        }
    }

    pub fn with_capacity(schema: Arc<Schema>, capacity: usize) -> Self {
        let p4s = (0..schema.n_p4s())
            .map(|_| Vec::with_capacity(capacity))
            .collect();

        let scalars = (0..schema.n_scalars())
            .map(|_| Vec::with_capacity(capacity))
            .collect();

        Self {
            schema,
            p4s,
            scalars,
            weights: None,
            len: 0,
        }
    }

    pub fn push<P, S>(&mut self, p4s: P, scalars: S) -> LadduDataResult<&mut Self>
    where
        P: IntoIterator<Item = RealVec4>,
        S: IntoIterator<Item = f64>,
    {
        self.push_event(OwnedEvent::new(
            p4s.into_iter().collect(),
            scalars.into_iter().collect(),
        ))
    }

    pub fn push_weighted<P, S>(
        &mut self,
        p4s: P,
        scalars: S,
        weight: f64,
    ) -> LadduDataResult<&mut Self>
    where
        P: IntoIterator<Item = RealVec4>,
        S: IntoIterator<Item = f64>,
    {
        self.push_event(OwnedEvent::weighted(
            p4s.into_iter().collect(),
            scalars.into_iter().collect(),
            weight,
        ))
    }

    pub fn push_event(&mut self, event: OwnedEvent) -> LadduDataResult<&mut Self> {
        if event.p4s.len() != self.schema.n_p4s() {
            return Err(LadduDataError::Schema(
                "wrong number of event vec4 values".into(),
            ));
        }

        if event.scalars.len() != self.schema.n_scalars() {
            return Err(LadduDataError::Schema(
                "wrong number of event scalar values".into(),
            ));
        }

        match (&mut self.weights, event.weight) {
            (Some(weights), Some(weight)) => weights.push(weight),

            (Some(_), None) => {
                return Err(LadduDataError::InvalidArgument(
                    "cannot mix weighted and unweighted events in one batch",
                ));
            }

            (None, Some(weight)) if self.len == 0 => {
                self.weights = Some(vec![weight]);
            }

            (None, Some(_)) => {
                return Err(LadduDataError::InvalidArgument(
                    "cannot mix unweighted and weighted events in one batch",
                ));
            }

            (None, None) => {}
        }

        for (col, value) in event.p4s.into_iter().enumerate() {
            self.p4s[col].push(value);
        }

        for (col, value) in event.scalars.into_iter().enumerate() {
            self.scalars[col].push(value);
        }

        self.len += 1;

        Ok(self)
    }

    pub fn extend<I>(&mut self, events: I) -> LadduDataResult<&mut Self>
    where
        I: IntoIterator<Item = OwnedEvent>,
    {
        for event in events {
            self.push_event(event)?;
        }

        Ok(self)
    }

    pub fn finish(self) -> LadduDataResult<EventBatch> {
        let p4s = self.p4s.into_iter().map(Arc::from).collect();
        let scalars = self.scalars.into_iter().map(Arc::from).collect();
        let weights = self.weights.map(Arc::from);

        EventBatch::new(self.schema, p4s, scalars, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64) -> RealVec4 {
        RealVec4 {
            x,
            y: x + 0.1,
            z: x + 0.2,
            t: x + 0.3,
        }
    }

    fn schema_with_weight() -> Arc<Schema> {
        Arc::new(Schema::new(["p"], ["x"], true).unwrap())
    }

    fn weighted_batch(start: usize, len: usize) -> EventBatch {
        let schema = schema_with_weight();

        let events = (start..start + len)
            .map(|i| OwnedEvent::weighted(vec![v(i as f64)], vec![i as f64], 10.0 + i as f64));

        EventBatch::from_events(schema, events).unwrap()
    }

    fn scalar_values(batch: &EventBatch) -> Vec<f64> {
        batch.scalar_column(0).to_vec()
    }

    #[test]
    fn event_batch_rejects_shape_mismatches_and_builder_rejects_mixed_weights() {
        let schema = schema_with_weight();

        let bad_vec4_count = EventBatch::new(
            Arc::clone(&schema),
            vec![],
            vec![Arc::from([1.0, 2.0])],
            Some(Arc::from([1.0, 2.0])),
        );

        assert!(matches!(bad_vec4_count, Err(LadduDataError::Schema(_))));

        let bad_lengths = EventBatch::new(
            Arc::clone(&schema),
            vec![Arc::from([v(1.0), v(2.0)])],
            vec![Arc::from([1.0])],
            Some(Arc::from([1.0, 2.0])),
        );

        assert!(matches!(bad_lengths, Err(LadduDataError::Schema(_))));

        let mut builder = EventBatchBuilder::new(schema);
        builder.push([v(1.0)], [1.0]).unwrap();

        let mixed = builder.push_weighted([v(2.0)], [2.0], 2.0);
        assert!(matches!(mixed, Err(LadduDataError::InvalidArgument(_))));
    }

    #[test]
    fn select_slice_filter_reweight_and_concat_preserve_columns_and_weight_semantics() {
        let weighted = weighted_batch(0, 4);
        let selected = weighted.select(&[3, 1]);

        assert_eq!(scalar_values(&selected), vec![3.0, 1.0]);
        assert_eq!(selected.weights_column().unwrap(), &[13.0, 11.0]);
        assert_eq!(selected.p4_at(0, 0).x, 3.0);
        assert_eq!(selected.p4_at(0, 1).t, 1.3);

        let sliced = weighted.slice(1, 3);
        assert_eq!(scalar_values(&sliced), vec![1.0, 2.0]);
        assert_eq!(sliced.weights_column().unwrap(), &[11.0, 12.0]);

        let filtered = weighted.filter(|ev| ev.scalar(0) >= 2.0);
        assert_eq!(scalar_values(&filtered), vec![2.0, 3.0]);

        let reweighted = filtered.reweight(|i, w| w + 100.0 + i as f64);
        assert_eq!(reweighted.weights_column().unwrap(), &[112.0, 114.0]);

        let schema = schema_with_weight();

        let unweighted_with_weight_schema = EventBatch::from_events(
            Arc::clone(&schema),
            [
                OwnedEvent::new(vec![v(100.0)], vec![100.0]),
                OwnedEvent::new(vec![v(101.0)], vec![101.0]),
            ],
        )
        .unwrap();

        let weighted_tail = EventBatch::from_events(
            schema,
            [
                OwnedEvent::weighted(vec![v(200.0)], vec![200.0], 5.0),
                OwnedEvent::weighted(vec![v(201.0)], vec![201.0], 6.0),
            ],
        )
        .unwrap();

        let concatenated =
            EventBatch::concat(&[unweighted_with_weight_schema, weighted_tail]).unwrap();

        assert_eq!(
            scalar_values(&concatenated),
            vec![100.0, 101.0, 200.0, 201.0]
        );
        assert_eq!(
            concatenated.weights_column().unwrap(),
            &[1.0, 1.0, 5.0, 6.0]
        );
    }
}
