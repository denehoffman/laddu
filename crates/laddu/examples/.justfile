set working-directory := "../../.."

data_events := env_var_or_default("DATA_EVENTS", "10000")
normalization_events := env_var_or_default("NORMALIZATION_EVENTS", "100000")
projection_json := env_var_or_default("PROJECTION_JSON", "target/ksks-closure.json")
projection_plot := env_var_or_default("PROJECTION_PLOT", "target/ksks-closure.png")

# Generate pseudo-data and normalization MC, fit them, and write the projection JSON.
fit:
    cargo run --release -p laddu --example generate_and_fit_ksks --features generation,fit -- {{data_events}} {{normalization_events}} {{projection_json}}

# Render an existing projection JSON without rerunning the fit.
render:
    uv run crates/laddu/examples/plot_ksks.py {{projection_json}} --output {{projection_plot}}

# Run the complete closure and plotting workflow.
plot: fit
    uv run crates/laddu/examples/plot_ksks.py {{projection_json}} --output {{projection_plot}}

default: plot
