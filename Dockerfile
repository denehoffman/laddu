FROM ubuntu:latest

ARG MSRV=1.88.0
ARG PYTHON_VERSION=3.9
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=$CARGO_HOME/bin:/root/.local/bin:$PATH
ENV UV_LINK_MODE=copy

RUN apt update && apt install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    pkg-config \
    python3-dev \
    openmpi-bin \
    libopenmpi-dev \
    libclang-dev \
    rsync

RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain ${MSRV} -y --no-modify-path

RUN rustup component add clippy rustfmt
RUN cargo install --locked cargo-nextest
RUN cargo install --locked just

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
WORKDIR /work
RUN uv python install ${PYTHON_VERSION}
RUN uv tool install "maturin[patchelf]"
RUN uv tool install ruff
RUN uv tool install ty
