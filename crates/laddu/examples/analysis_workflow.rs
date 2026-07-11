use std::sync::Arc;

use laddu::{
    data::{data::OwnedEvent, schema::Schema},
    prelude::*,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["mass"], true)?);
    let dataset = Dataset::from_events(
        schema,
        [
            OwnedEvent::weighted(vec![], vec![1.1], 1.0),
            OwnedEvent::weighted(vec![], vec![1.5], 1.0),
            OwnedEvent::weighted(vec![], vec![1.9], 1.0),
        ],
    )?;
    let execution = Execution::default();
    let mass = event_scalar("mass");
    let selected = dataset.select(
        &Predicate::ge(mass.clone(), 1.2).and(Predicate::lt(mass.clone(), 2.0)),
        &execution,
    )?;
    let bins = selected.bin_by(&mass, BinSpec::uniform(2, 1.2, 2.0)?, &execution)?;

    let s_wave = Expr::from(parameter!("S", initial: 1.0)).tagged("S");
    let d_wave = (Expr::from(parameter!("D", initial: 0.5)) * mass).tagged("D");
    let model = CompiledModel::from_expr(&(s_wave + d_wave).norm_sqr())?;
    let likelihood = Likelihood::new([
        Box::new(NllTerm::new("waves", &model, &selected, &selected)?) as Box<dyn LikelihoodTerm>,
    ])?;
    let projection = likelihood.projection("waves", &selected, ["S"])?;
    let weights = projection.weights(&likelihood.default_params(), true)?;

    println!("{} bins, {} projected weights", bins.len(), weights.len());
    Ok(())
}
