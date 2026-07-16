set working-directory := "../../.."

data_events := env_var_or_default("DATA_EVENTS", "10000")
normalization_events := env_var_or_default("NORMALIZATION_EVENTS", "100000")
projection_json := env_var_or_default("PROJECTION_JSON", "target/ksks-closure.json")
projection_plot := env_var_or_default("PROJECTION_PLOT", "target/ksks-closure.png")
backend := env_var_or_default("BACKEND", "cpu")
features := if backend == "jit" { "generation,fit,jit" } else if backend == "gpu" { "generation,fit,wgpu" } else { "generation,fit" }

# Generate pseudo-data and normalization MC, fit them, and write the projection JSON.
fit:
    cargo run --release -p laddu --example generate_and_fit_ksks --features {{features}} -- {{data_events}} {{normalization_events}} {{projection_json}} {{backend}}

# Build the selected backend without running the closure.
build:
    cargo build --release -p laddu --example generate_and_fit_ksks --features {{features}}

# Time only the already-built closure executable; the build dependency is not timed.
time: build
    time -p target/release/examples/generate_and_fit_ksks {{data_events}} {{normalization_events}} {{projection_json}} {{backend}}

# Render an existing projection JSON without rerunning the fit.
render:
    uv run crates/laddu/examples/plot_ksks.py {{projection_json}} --output {{projection_plot}}

# Run the complete closure and plotting workflow.
plot: fit
    uv run crates/laddu/examples/plot_ksks.py {{projection_json}} --output {{projection_plot}}

default: plot
