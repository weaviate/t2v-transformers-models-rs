# t2v-transformers-models-rs

This repository contains the Proof of Concept (PoC) for a Rust-native sentence embedding API for use within Weaviate's module vectorization framework.

In its current form, it uses [huggingface/candle](https://github.com/huggingface/candle/tree/main) to load HuggingFace models from the Hub into Rust. The models are then used to generate embeddings for input text.

The API layer is provided by the [tokio-rs/axum](https://github.com/tokio-rs/axum) framework, which allows for event loop based concurrency.

The inference workloads themselves are scheduled into a rayon thread-pool using [andybarron/tokio-rayon](https://github.com/andybarron/tokio-rayon), which is a lightweight wrapper allowing for awaiting of `rayon` jobs within a `tokio` async function. Ideally, this would be re-implemented by us internally to avoid the dangerous external dependency. For now, it is used as a proof of concept.

## Motivation

In terms of raw performance, when working a purely serial single-threaded runtime whereby the server responds to only one request at a time, both the `rs` and `py` implementations are comparable in speed. However, when running concurrently, the `rs` implementation never slows down due to the rust-native implementation of `candle` and the __blazingly fast__ behaviour of `tokio` & `rayon`. On the other hand, the `py` implementation suffers significantly in a concurrent workload due to Python's GIL blocking parallel inference on the model.