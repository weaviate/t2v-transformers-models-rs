# Use just rust for the CPU build
FROM rust:1.78 as cpu-sys

WORKDIR /app

# Copy our manifests
COPY ./Cargo.lock ./Cargo.lock
COPY ./Cargo.toml ./Cargo.toml
COPY ./src ./src

# Use the NVIDIA CUDA base image for the GPU build
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 as gpu-sys

# Install system dependencies
RUN apt-get update -qq \
    && apt-get install -qq -y vim gcc g++ curl git build-essential libssl-dev openssl pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy our manifests
COPY ./Cargo.lock ./Cargo.lock
COPY ./Cargo.toml ./Cargo.toml
COPY ./src ./src

FROM cpu-sys as cpu-build
# Build only the dependencies to cache them
RUN cargo build --release

FROM gpu-sys as gpu-build
# Build only the dependencies to cache them
RUN cargo add candle-core --features "cuda"
RUN cargo build --release

FROM rust:1.78.0-alpine3.20 as cpu-release
COPY --from=cpu-build /app/target/release/bin /usr/local/bin/t2v-rs
ENV ENABLE_CUDA false
ENTRYPOINT ["/usr/local/bin/t2v-rs"]

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 as gpu-release
COPY --from=gpu-build /app/target/release/bin /usr/local/bin/t2v-rs
ENV ENABLE_CUDA true
ENTRYPOINT ["/usr/local/bin/t2v-rs"]