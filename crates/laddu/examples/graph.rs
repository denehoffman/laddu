use std::error::Error;

use laddu::parameter;
use laddu_physics::math::{SphericalHarmonic, WignerDMatrix};

fn main() -> Result<(), Box<dyn Error>> {
    let w = WignerDMatrix::new(2, 1, -1)?;
    let y = SphericalHarmonic::new(4, 2)?;
    let costheta = parameter!("costheta");
    let phi = parameter!("phi");
    let model = (w.D(&costheta, &phi, -&phi) * y.evaluate(&costheta, &phi)).norm_sqr();
    println!("{}", model.to_graph());
    Ok(())
}
