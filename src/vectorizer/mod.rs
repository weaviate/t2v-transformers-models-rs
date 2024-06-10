pub mod hf;

use serde::{Deserialize, Serialize};
use candle_core::{backend::BackendDevice, CudaDevice, Device};

#[derive(Deserialize)]
pub struct VectorInputConfig {
    pooling_strategy: String,
}

#[derive(Clone, Serialize)]
pub struct Meta {
    model_type: String,
}

pub trait Vectorizer: Send + Sync {
    fn get_meta(&self) -> Meta;
    fn vectorize(
        &self,
        texts: Vec<String>,
        config: VectorInputConfig,
    ) -> Result<Vec<Vec<f64>>, &'static str>;
}

pub struct VectorizerConfig {
    device: Device,
    use_direct_tokenize: bool,
}

impl VectorInputConfig {
    pub fn new(pooling_strategy: String) -> Self {
        VectorInputConfig { pooling_strategy }
    }
}

impl VectorizerConfig {
    pub fn new(cuda_core: String, use_direct_tokenize: bool, use_cuda: bool) -> Self {
        let device = match use_cuda {
            true => {
                let core = match cuda_core.split(":").collect::<Vec<&str>>().as_slice() {
                    [_, core] => core.parse::<usize>().unwrap(),
                    _ => panic!(
                        "Invalid CUDA core: {}. Use the `cuda:n` convention instead.",
                        cuda_core
                    ),
                };
                let device = match CudaDevice::new(core) {
                    Ok(device) => device,
                    Err(e) => panic!("Invalid CUDA device: {}", e),
                };
                Device::Cuda(device)
            }
            false => Device::Cpu,
        };
        VectorizerConfig {
            device,
            use_direct_tokenize,
        }
    }
}

