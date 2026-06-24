use std::error::Error;

use laddu::compile::CompiledModel;
use laddu::parameter;
use laddu_physics::math::{SphericalHarmonic, WignerDMatrix};

fn main() -> Result<(), Box<dyn Error>> {
    let w = WignerDMatrix::new(2, 1, -1)?;
    let y = SphericalHarmonic::new(4, 2)?;
    let costheta = parameter!("costheta");
    let phi = parameter!("phi");
    let model = (w.D(&costheta, &phi, -&phi) * y.evaluate(&costheta, &phi)).norm_sqr();
    println!(
        "Raw graph:\n{}\n\n{}",
        model.to_graph().display_tree(),
        model.to_graph()
    );

    let compiled = CompiledModel::from_expr(&model)?;
    println!(
        "Optimized graph:\n{}\n\n{}",
        compiled.graph().display_tree(),
        compiled.graph()
    );
    Ok(())
}
