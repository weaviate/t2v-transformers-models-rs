mod vectorizer;
use crate::{
    vectorizer::shared::{
        get_config, get_graph, get_tokenizer, get_weights, Meta, VectorInputConfig, Vectorize,
        VectorizerConfig,
    },
    vectorizer::{candle::CandleBert, onnx::OnnxBert},
};

use std::env;
use std::sync::Arc;

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
    taken: f64,
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
    let start = std::time::Instant::now();
    let vector = tokio_rayon::spawn(move || {
        let vec = match state.vectorizer.vectorize(vec![text], config) {
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
                    taken: start.elapsed().as_secs_f64(),
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
    vectorizer: Arc<Box<dyn Vectorize>>,
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
    let direct_tokenize = match env::var("T2V_TRANSFORMERS_DIRECT_TOKENIZE") {
        Ok(direct_tokenize) => direct_tokenize == "true" || direct_tokenize == "1",
        Err(_) => false,
    };
    let cuda_support = match env::var("ENABLE_CUDA") {
        Ok(cuda_env) => cuda_env == "true" || cuda_env == "1",
        Err(_) => false,
    };
    let tokenizer = match env::var("TOKENIZER_PATH") {
        Ok(tokenizer_filename) => std::path::PathBuf::from(tokenizer_filename.to_string()),
        Err(_) => match env::var("HF_MODEL_ID") {
            Ok(model) => get_tokenizer(model),
            Err(_) => panic!("Neither TOKENIZER_PATH nor HF_MODEL_ID is set"),
        },
    };
    let model = match env::var("MODEL_PATH") {
        Ok(model_filename) => std::path::PathBuf::from(model_filename.to_string()),
        Err(_) => match env::var("HF_MODEL_ID") {
            Ok(model) => match cuda_support {
                true => get_weights(model),
                false => get_graph(model),
            },
            Err(_) => panic!("Neither MODEL_PATH nor HF_MODEL_ID is set"),
        },
    };
    let config = match cuda_support {
        true => match env::var("CONFIG_PATH") {
            Ok(config_filename) => Some(std::path::PathBuf::from(config_filename.to_string())),
            Err(_) => match env::var("HF_MODEL_ID") {
                Ok(model) => Some(get_config(model)),
                Err(_) => panic!("Neither CONFIG_PATH nor HF_MODEL_ID is set"),
            },
        },
        false => None,
    };

    let vectorizer: Box<dyn Vectorize> = match cuda_support {
        true => Box::new(CandleBert::new(
            model,
            tokenizer,
            config.unwrap(),
            VectorizerConfig::new(direct_tokenize),
        )),
        false => Box::new(OnnxBert::new(
            model,
            tokenizer,
            VectorizerConfig::new(direct_tokenize),
        )),
    };

    let app_state = AppState {
        meta: vectorizer.get_meta(),
        vectorizer: Arc::new(vectorizer),
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
