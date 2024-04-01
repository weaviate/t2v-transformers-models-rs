use std::sync::{Arc,Mutex};

use axum::{
    extract::State, http::StatusCode, routing::post, Json, Router
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
    fn vectorize(& mut self, texts: Vec<String>) -> Vec<Vec<f32>>;
}

struct CandleBert {
    model: BertModel,
    tokenizer: Tokenizer
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
        let config = std::fs::read_to_string(config_filename).unwrap();
        let mut config: Config = serde_json::from_str(&config).unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &Device::Cpu).unwrap()
        };
        config.hidden_act = HiddenAct::GeluApproximate;
        let model = BertModel::load(vb, &config).unwrap();
        CandleBert{model, tokenizer}
    }
}

// impl Vectorizer for RustBert {
//     fn vectorize(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
//         self.model.encode(&texts).unwrap()
//     }
// }

impl Vectorizer for CandleBert {
    fn vectorize(& mut self, texts: Vec<String>) -> Vec<Vec<f32>> {
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }
        let tokens = self.tokenizer
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
        };
        out
    }
}

async fn vectors(
    State(state): State<AppState>,
    Json(payload): Json<VectorsInput>
) -> (StatusCode, Json<VectorsOutput>) {
    (
        StatusCode::OK,
        Json(VectorsOutput {
            vectors: tokio_rayon::spawn(move || {
                state.vectorizer.clone().lock().unwrap().vectorize(payload.texts)
            }).await,
        })
    )
}

#[derive(Clone)]
struct AppState {
    vectorizer: Arc<Mutex<dyn Vectorizer>>
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // let vectorizer = Arc::new(Mutex::new(RustBert::new(SentenceEmbeddingsModelType::AllMiniLmL6V2)));
    let vectorizer = Arc::new(Mutex::new(CandleBert::new(
        "BAAI/bge-small-en-v1.5".to_string(),
        "refs/pr/9".to_string()
    )));

    let app_state = AppState { vectorizer };

    let app = Router::new()
        .route("/vectors", post(vectors))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}