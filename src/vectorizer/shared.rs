use hf_hub::{api::sync::Api, Repo, RepoType};
use log::info;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct VectorInputConfig {
    pub pooling_strategy: String,
}

#[derive(Clone, Serialize)]
pub struct Meta {
    pub model_type: String,
}

pub trait Vectorize: Send + Sync {
    fn get_meta(&self) -> Meta;
    fn vectorize(
        &self,
        texts: Vec<String>,
        config: VectorInputConfig,
    ) -> Result<Vec<Vec<f64>>, &'static str>;
}

pub struct VectorizerConfig {
    use_direct_tokenize: bool,
}

impl VectorInputConfig {
    pub fn new(pooling_strategy: String) -> Self {
        VectorInputConfig { pooling_strategy }
    }
}

impl VectorizerConfig {
    pub fn new(use_direct_tokenize: bool) -> Self {
        VectorizerConfig {
            use_direct_tokenize,
        }
    }
}

pub fn get_weights(model: String) -> std::path::PathBuf {
    let api = Api::new().unwrap().repo(Repo::new(model, RepoType::Model));
    let weights = api.get("model.safetensors").unwrap();
    info!("downloaded weights to {}", weights.to_str().unwrap());
    weights
}

pub fn get_graph(model: String) -> std::path::PathBuf {
    let api = Api::new().unwrap().repo(Repo::new(model, RepoType::Model));
    let graph = api.get("onnx/model.onnx").unwrap();
    info!("downloaded graph to {}", graph.to_str().unwrap());
    graph
}

pub fn get_tokenizer(model: String) -> std::path::PathBuf {
    let api = Api::new().unwrap().repo(Repo::new(model, RepoType::Model));
    let tokenizer = api.get("tokenizer.json").unwrap();
    info!("downloaded tokenizer to {}", tokenizer.to_str().unwrap());
    tokenizer
}

pub fn get_config(model: String) -> std::path::PathBuf {
    let api = Api::new().unwrap().repo(Repo::new(model, RepoType::Model));
    let config = api.get("config.json").unwrap();
    info!("downloaded config to {}", config.to_str().unwrap());
    config
}
