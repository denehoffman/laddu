FROM ubuntu:24.04

ARG RUST_VERSION=1.94.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        just \
        libclang-dev \
        libopenmpi-dev \
        libssl-dev \
        openmpi-bin \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- --default-toolchain "${RUST_VERSION}" --profile minimal -y \
    && rustup component add clippy rustfmt

ADD https://astral.sh/uv/install.sh /tmp/uv-installer.sh
RUN sh /tmp/uv-installer.sh \
    && rm /tmp/uv-installer.sh \
    && uv python install 3.11 \
    && uv tool install prek

RUN cargo install --locked cargo-hack cargo-nextest
