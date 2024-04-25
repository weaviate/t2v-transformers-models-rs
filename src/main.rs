use std::sync::{Arc, RwLock};

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

use tokio_rayon;

#[derive(Deserialize)]
struct VectorInputConfig {
    pooling_strategy: String,
}

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
    vector: Vec<f32>,
    dim: usize,
}

#[derive(Serialize)]
struct VectorsOutput {
    vectors: Vec<Vec<f32>>,
}

trait Vectorizer: Send + Sync {
    fn get_meta(&self) -> Meta;
    fn vectorize(&self, texts: Vec<String>) -> Vec<Vec<f32>>;
}

struct CandleBert {
    meta: Meta,
    model: BertModel,
    tokenizer: Tokenizer,
}

#[derive(Clone, Serialize)]
struct Meta {
    model_type: String,
}

#[derive(Serialize)]
struct MetaOutput {
    model: Meta,
}

impl CandleBert {
    fn new(model_id: String, revision: String) -> Self {
        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new().unwrap();
            let api = api.repo(repo);
            let config = api.get("config.json").unwrap();
            let tokenizer = api.get("tokenizer.json").unwrap();
            let weights = api.get("model.safetensors").unwrap();
            (config, tokenizer, weights)
        };
        let config_str = std::fs::read_to_string(config_filename).unwrap();
        let mut config: Config = serde_json::from_str(&config_str).unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &Device::Cpu).unwrap()
        };
        config.hidden_act = HiddenAct::GeluApproximate;
        let model = BertModel::load(vb, &config).unwrap();
        CandleBert {
            meta: Meta {
                model_type: "candle-bert".to_owned(),
            },
            model,
            tokenizer,
        }
    }
}

// impl Vectorizer for RustBert {
//     fn vectorize(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
//         self.model.encode(&texts).unwrap()
//     }
// }

impl Vectorizer for CandleBert {
    fn get_meta(&self) -> Meta {
        self.meta.clone()
    }
    fn vectorize(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        let mut tokenizer = self.tokenizer.clone();
        let tokens = tokenizer
            .with_padding(Some({
                PaddingParams {
                    strategy: tokenizers::PaddingStrategy::BatchLongest,
                    ..Default::default()
                }
            }))
            .encode_batch(texts.clone(), true)
            .unwrap();
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), &Device::Cpu).unwrap()
            })
            .collect::<Vec<_>>();
        let token_ids = Tensor::stack(&token_ids, 0).unwrap();
        let token_type_ids = token_ids.zeros_like().unwrap();
        println!("running inference on batch {:?}", token_ids.shape());
        let embeddings = self.model.forward(&token_ids, &token_type_ids).unwrap();
        println!("generated embeddings {:?}", embeddings.shape());
        let mut out = Vec::<Vec<f32>>::new();
        for i in 0..texts.len() {
            let e_i = embeddings.get(0).unwrap().get(i).unwrap();
            out.push(e_i.to_vec1().unwrap());
        }
        out
    }
}

async fn vectors(
    State(state): State<AppState>,
    Json(payload): Json<VectorInput>,
) -> (StatusCode, Json<VectorOutput>) {
    let text = payload.text.clone();
    let vector = tokio_rayon::spawn(move || {
        let vectorizer = state.vectorizer.read().unwrap(); // Read lock for concurrent reads
        let vec = vectorizer.vectorize(vec![text]).pop().unwrap();
        vec
    })
    .await;
    let vector_len = vector.len();
    (
        StatusCode::OK,
        Json(VectorOutput {
            text: payload.text,
            vector: vector,
            dim: vector_len,
        }),
    )
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
    // let vectorizer = Arc::new(Mutex::new(RustBert::new(SentenceEmbeddingsModelType::AllMiniLmL6V2)));
    let vectorizer = CandleBert::new(
        "BAAI/bge-small-en-v1.5".to_string(),
        "refs/pr/9".to_string(),
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

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}
