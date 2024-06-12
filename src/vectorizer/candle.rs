use crate::vectorizer::shared::{Meta, VectorInputConfig, Vectorize, VectorizerConfig};
use log::info;
use std::env;
use tracing::error;

use candle_core::{backend::BackendDevice, CudaDevice, DType, Device, Tensor};
use candle_nn::{
    var_builder::{SimpleBackend, VarBuilderArgs},
    VarBuilder,
};
use candle_transformers::models::bert::{BertModel, Config, HiddenAct};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{Encoding, PaddingParams, Tokenizer};

pub struct CandleBert {
    config: VectorizerConfig,
    device: Device,
    meta: Meta,
    model: BertModel,
    tokenizer: Tokenizer,
}

impl CandleBert {
    fn get_device() -> Device {
        match env::var("CUDA_CORE") {
            Ok(cuda_core) => match cuda_core.as_str() {
                "" => Device::Cuda(CudaDevice::new(0).unwrap()),
                _ => {
                    let core = match cuda_core.split(":").collect::<Vec<&str>>().as_slice() {
                        [_, core] => core.parse::<usize>().unwrap(),
                        _ => panic!(
                            "Invalid CUDA core: {}. Use the `cuda:n` convention instead.",
                            cuda_core
                        ),
                    };
                    Device::Cuda(CudaDevice::new(core).unwrap())
                }
            },
            Err(_) => Device::Cuda(CudaDevice::new(0).unwrap()),
        }
    }
    pub fn new(
        model_id: String,
        revision: Option<String>,
        vectorizer_config: VectorizerConfig,
    ) -> Self {
        let device = Self::get_device(); // detect CUDA core from env
        let repo = match revision {
            Some(revision) => {
                info!(
                    "loading model {} with revision {}",
                    model_id.clone(),
                    revision.clone()
                );
                Repo::with_revision(model_id, RepoType::Model, revision)
            }
            None => {
                info!(
                    "loading model {} with no specified revision",
                    model_id.clone()
                );
                Repo::new(model_id, RepoType::Model)
            }
        };
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new().unwrap();
            let api = api.repo(repo);

            let config = api.get("config.json").unwrap();
            let tokenizer = api.get("tokenizer.json").unwrap();
            let weights = api.get("model.safetensors").unwrap();

            info!("downloaded config to {}", config.to_str().unwrap());
            info!("downloaded tokenizer to {}", tokenizer.to_str().unwrap());
            info!("downloaded weights to {}", weights.to_str().unwrap());

            (config, tokenizer, weights)
        };
        let config_str = std::fs::read_to_string(config_filename).unwrap();
        let mut config: Config = serde_json::from_str(&config_str).unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

        let vb: VarBuilderArgs<Box<dyn SimpleBackend>> = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F64, &device).unwrap()
        };
        config.hidden_act = HiddenAct::GeluApproximate;
        let model = BertModel::load(vb, &config).unwrap();
        CandleBert {
            config: vectorizer_config,
            device: device,
            meta: Meta {
                model_type: "candle-bert".to_owned(),
            },
            model,
            tokenizer,
        }
    }
    fn pool_embeddings(
        &self,
        embeddings: &Tensor,
        attention_mask: &Tensor,
        pooling_strategy: String,
    ) -> Result<Box<Tensor>, &'static str> {
        match pooling_strategy.as_str() {
            "cls" => self.pool_get(embeddings),
            "masked_mean" => self.pool_sum(embeddings, attention_mask),
            _ => panic!("Pooling strategy {} not implemented", pooling_strategy),
        }
    }
    fn pool_get(&self, embeddings: &Tensor) -> Result<Box<Tensor>, &'static str> {
        match embeddings.get_on_dim(1, 0) {
            Ok(vectors) => match vectors.sum(0) {
                Ok(vectors) => Ok(Box::new(vectors)),
                Err(e) => {
                    error!("error summing vectors: {:?}", e);
                    return Err("error summing vectors");
                }
            },
            Err(e) => {
                error!("error pooling embeddings: {:?}", e);
                return Err("error pooling embeddings");
            }
        }
    }
    fn get_sum_embeddings_mask(
        &self,
        embeddings: &Tensor,
        input_mask_expanded: &Tensor,
    ) -> Result<(Tensor, Tensor), &'static str> {
        let sum_embeddings = match embeddings * input_mask_expanded {
            Ok(mult_res) => match mult_res.sum(1) {
                Ok(sum_embeddings) => sum_embeddings,
                Err(e) => {
                    error!("error summing embeddings: {:?}", e);
                    return Err("error summing embeddings");
                }
            },
            Err(e) => {
                error!("error multiplying embeddings: {:?}", e);
                return Err("error multiplying embeddings");
            }
        };
        let sum_mask = match input_mask_expanded.sum(1) {
            Ok(sum_mask) => match sum_mask.clamp(1e-9, f64::INFINITY) {
                Ok(sum_mask) => sum_mask,
                Err(e) => {
                    error!("error clamping mask: {:?}", e);
                    return Err("error clamping mask");
                }
            },
            Err(e) => {
                error!("error summing mask: {:?}", e);
                return Err("error summing mask");
            }
        };
        Ok((sum_embeddings, sum_mask))
    }
    fn pool_sum(
        &self,
        embedding: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Box<Tensor>, &'static str> {
        let mask = match attention_mask.unsqueeze(attention_mask.dims().len()) {
            Ok(mask) => match mask.expand(embedding.shape()) {
                Ok(mask) => match mask.to_dtype(DType::F64) {
                    Ok(mask) => mask,
                    Err(e) => {
                        error!("error casting attention mask to dtype: {:?}", e);
                        return Err("error casting attention mask to dtype");
                    }
                },
                Err(e) => {
                    error!("error expanding attention mask: {:?}", e);
                    return Err("error expanding attention mask");
                }
            },
            Err(e) => {
                error!("error unsqueezing attention mask: {:?}", e);
                return Err("error unsqueezing attention mask");
            }
        };

        let (sum_embeddings, sum_mask) = match self.get_sum_embeddings_mask(embedding, &mask) {
            Ok((sum_embeddings, sum_mask)) => (sum_embeddings, sum_mask),
            Err(e) => {
                error!("error getting sum embeddings and mask: {:?}", e);
                return Err("error getting sum embeddings and mask");
            }
        };

        let sentences = match sum_embeddings / sum_mask {
            Ok(sentences) => sentences,
            Err(e) => {
                error!("error dividing sum embeddings by sum mask: {:?}", e);
                return Err("error dividing sum embeddings by sum mask");
            }
        };

        match sentences.sum(0) {
            Ok(sentences) => Ok(Box::new(sentences)),
            Err(e) => {
                error!("error summing sentences: {:?}", e);
                return Err("error summing sentences");
            }
        }
    }
    fn get_tensors_from_encodings(
        &self,
        encodings: Vec<Encoding>,
    ) -> Result<(Tensor, Tensor, Tensor), &'static str> {
        let mut attention_mask_vec = Vec::<Tensor>::new();
        let mut token_ids_vec = Vec::<Tensor>::new();
        for encoding in encodings {
            let attention_mask = match Tensor::new(
                encoding.get_attention_mask().to_vec().as_slice(),
                &self.device,
            ) {
                Ok(attention_mask) => attention_mask,
                Err(e) => {
                    error!("error creating attention mask tensor: {:?}", e);
                    return Err("error creating attention mask tensor");
                }
            };
            let token_ids = match Tensor::new(encoding.get_ids().to_vec().as_slice(), &self.device)
            {
                Ok(token_ids) => token_ids,
                Err(e) => {
                    error!("error creating token ids tensor: {:?}", e);
                    return Err("error creating token ids tensor");
                }
            };
            attention_mask_vec.push(attention_mask);
            token_ids_vec.push(token_ids);
        }

        let attention_mask = match Tensor::stack(&attention_mask_vec, 0) {
            Ok(attention_mask) => attention_mask,
            Err(e) => {
                error!("error stacking attention mask tensors: {:?}", e);
                return Err("error stacking attention mask tensors");
            }
        };
        let token_ids = match Tensor::stack(&token_ids_vec, 0) {
            Ok(token_ids) => token_ids,
            Err(e) => {
                error!("error stacking token ids tensors: {:?}", e);
                return Err("error stacking token ids tensors");
            }
        };
        let token_type_ids = match token_ids.zeros_like() {
            Ok(token_type_ids) => token_type_ids,
            Err(e) => {
                error!("error creating token type ids tensor: {:?}", e);
                return Err("error creating token type ids tensor");
            }
        };
        Ok((attention_mask, token_ids, token_type_ids))
    }
    fn tokenize(&self, texts: Vec<String>) -> Result<Vec<Encoding>, &'static str> {
        match self
            .tokenizer
            .clone()
            .with_padding(Some({
                PaddingParams {
                    strategy: tokenizers::PaddingStrategy::BatchLongest,
                    ..Default::default()
                }
            }))
            .encode_batch(texts.clone(), true)
        {
            Ok(encodings) => Ok(encodings),
            Err(e) => {
                error!("error tokenizing batch: {:?}", e);
                return Err("error tokenizing batch");
            }
        }
    }
}

impl Vectorize for CandleBert {
    fn get_meta(&self) -> Meta {
        self.meta.clone()
    }
    fn vectorize(
        &self,
        texts: Vec<String>,
        config: VectorInputConfig,
    ) -> Result<Vec<Vec<f64>>, &'static str> {
        let encodings = self.tokenize(texts)?;
        let (attention_mask, token_ids, token_type_ids) =
            self.get_tensors_from_encodings(encodings)?;

        info!("running inference on batch {:?}", token_ids.shape());
        let embeddings = match self.model.forward(&token_ids, &token_type_ids) {
            Ok(embeddings) => embeddings,
            Err(e) => {
                error!("error running inference: {:?}", e);
                return Err("error running inference");
            }
        };
        info!("generated embeddings {:?}", embeddings.shape());

        let vectors =
            self.pool_embeddings(&embeddings, &attention_mask, config.pooling_strategy)?;
        let out = match vectors.to_vec1::<f64>() {
            Ok(v) => v,
            Err(e) => {
                error!("error converting vectors to vec: {:?}", e);
                return Err("error converting vectors to vec");
            }
        };
        Ok(vec![out])
    }
}
