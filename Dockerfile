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
    rsync

# Install mise
RUN curl https://mise.run | sh
ENV PATH="/root/.local/bin:/root/.mise/bin:${PATH}"

# Preinstall toolchain dependencies
RUN mise use --yes rust@latest \
    && mise use --yes cargo@latest \
    && mise use --yes cargo:cargo-nextest@latest \
    && mise use --yes cargo:cargo-hack@latest \
    && mise use --yes uv@latest
