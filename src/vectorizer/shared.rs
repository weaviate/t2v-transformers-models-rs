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
