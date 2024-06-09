mod vec;
use crate::vec::{CandleBert, Meta, VectorInputConfig, Vectorizer};

use std::env;
use std::sync::{Arc, RwLock};

use axum::response::{IntoResponse, Response};
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use log::info;
use serde::{Deserialize, Serialize};

use tokio_rayon;
use vec::VectorizerConfig;

#[derive(Deserialize)]
struct VectorInput {
    text: String,
    config: Option<VectorInputConfig>,
}

#[derive(Deserialize)]
struct VectorsInput {
    texts: Vec<String>,
    config: Option<VectorInputConfig>,
}

#[derive(Serialize)]
struct VectorOutput {
    text: String,
    vector: Vec<f64>,
    dim: usize,
}

#[derive(Serialize)]
struct VectorsOutput {
    vectors: Vec<Vec<f64>>,
}

#[derive(Serialize)]
struct MetaOutput {
    model: Meta,
}

async fn vectors(State(state): State<AppState>, Json(payload): Json<VectorInput>) -> Response {
    let text = payload.text.clone();
    let config = payload
        .config
        .unwrap_or(VectorInputConfig::new("masked_mean".to_string()));
    let vector = tokio_rayon::spawn(move || {
        let vectorizer = match state.vectorizer.read() {
            Ok(v) => v,
            Err(_) => return Err("Failed to acquire read lock".to_string()),
        }; // Read lock for concurrent reads of shared vectorizer memory
        let vec = match vectorizer.vectorize(vec![text], config) {
            Ok(mut vec) => match vec.pop() {
                Some(v) => Ok(v),
                None => Err("Failed to get vector".to_string()),
            },
            Err(e) => Err(e.to_string()),
        };
        vec
    })
    .await;
    match vector {
        Ok(vector) => {
            let vector_len = vector.len();
            (
                StatusCode::OK,
                Json(VectorOutput {
                    text: payload.text,
                    vector,
                    dim: vector_len,
                }),
            )
                .into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

async fn live() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn ready() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn meta(State(state): State<AppState>) -> (StatusCode, Json<MetaOutput>) {
    (StatusCode::OK, Json(MetaOutput { model: state.meta }))
}

#[derive(Clone)]
struct AppState {
    meta: Meta,
    vectorizer: Arc<RwLock<dyn Vectorizer>>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let port = match env::var("PORT") {
        Ok(port) => port,
        Err(_) => "3000".to_string(),
    };
    let model = match env::var("HF_MODEL_ID") {
        Ok(model) => model,
        Err(_) => panic!("HF_MODEL_ID is not set"),
    };
    let revision = match env::var("HF_MODEL_REVISION") {
        Ok(revision) => revision,
        Err(_) => panic!("HF_MODEL_REVISION is not set"),
    };
    let direct_tokenize = match env::var("T2V_TRANSFORMERS_DIRECT_TOKENIZE") {
        Ok(direct_tokenize) => direct_tokenize == "true" || direct_tokenize == "1",
        Err(_) => false,
    };
    let cuda_support = match env::var("ENABLE_CUDA") {
        Ok(cuda_env) => cuda_env == "true" || cuda_env == "1",
        Err(_) => false,
    };
    let cuda_core = match cuda_support {
        true => match env::var("CUDA_CORE") {
            Ok(cuda_core) => match cuda_core.as_str() {
                "" => "cuda:0".to_string(),
                _ => cuda_core,
            },
            Err(_) => "cuda:0".to_string(),
        },
        false => "".to_string(),
    };

    let vectorizer = CandleBert::new(
        model,
        revision,
        VectorizerConfig::new(cuda_core, direct_tokenize, cuda_support),
    );

    let app_state = AppState {
        meta: vectorizer.get_meta(),
        vectorizer: Arc::new(RwLock::new(vectorizer)),
    };

    let app = Router::new()
        .route("/vectors", post(vectors))
        .route("/vectors/", post(vectors))
        .route("/.well-known/live", get(live))
        .route("/.well-known/ready", get(ready))
        .route("/meta", get(meta))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .unwrap();
    info!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}
