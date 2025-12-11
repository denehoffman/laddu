FROM ubuntu:rolling

RUN apt update && apt install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    python3-dev \
    openmpi-bin \
    libopenmpi-dev \
    libclang-dev \
    libssl-dev \
    rsync \
    just


RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup component add clippy rustfmt

RUN cargo install --locked cargo-nextest
RUN cargo install --locked cargo-hack

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
