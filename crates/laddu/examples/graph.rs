//! Builds, compiles, and displays a small tagged expression graph.

use std::error::Error;

use laddu::compile::CompiledModel;
use laddu::{ColorPreset, DisplayColor, NodeSelector, NodeStyle, NodeStyleRule, parameter};
use laddu_physics::math::{WignerDMatrix, spherical_harmonic};

fn main() -> Result<(), Box<dyn Error>> {
    let w = WignerDMatrix::new(2, 1, -1)?;
    let costheta = parameter!("costheta");
    let phi = parameter!("phi");
    let model =
        (w.D(&costheta, &phi, -&phi) * spherical_harmonic(4, 2, &costheta, &phi)?).norm_sqr();
    let raw = model.to_graph();
    println!("Raw graph:\n{}\n\n{}", raw.display_tree(), raw);

    let compiled = CompiledModel::from_expr(&model)?;
    println!(
        "Optimized graph:\n{}\n\n{}",
        compiled.display_tree().with_preset(ColorPreset::Dark),
        compiled.graph()
    );

    let highlighted = NodeStyleRule::new(
        NodeSelector::Name("costheta".to_owned()),
        NodeStyle::new().with_foreground(DisplayColor::rgb(255, 128, 0)),
    );
    let dot = compiled
        .display_dot()
        .expand_repeated(false)
        .with_preset(ColorPreset::Light)
        .with_style_rule(highlighted);
    println!("Optimized graph as compact, colored DOT:\n{dot}");

    #[cfg(feature = "svg")]
    std::fs::write("model.svg", dot.render_svg()?)?;

    Ok(())
}
