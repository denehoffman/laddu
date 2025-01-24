#!/bin/bash

source /opt/venv/bin/activate

uv pip install maturin[patchelf]

# Install laddu
maturin develop --uv -m py-laddu/Cargo.toml

cd benchmark

/bin/bash
