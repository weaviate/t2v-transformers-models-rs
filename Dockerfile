# Rust as the base image
FROM rust:1.77 as build

WORKDIR /app

# 2. Copy our manifests
COPY ./Cargo.lock ./Cargo.lock
COPY ./Cargo.toml ./Cargo.toml
COPY ./src ./src

# 3. Build only the dependencies to cache them
RUN cargo build --release

FROM rust:1.77 as release

# 5. Build for release.
WORKDIR /t2v-rs
COPY --from=build /app/target/release/bin /usr/local/bin/t2v-rs

ENTRYPOINT ["/usr/local/bin/t2v-rs"]