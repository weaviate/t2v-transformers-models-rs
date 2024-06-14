use std::{iter::zip, path::PathBuf};

use crate::vectorizer::shared::{Meta, VectorInputConfig, Vectorize, VectorizerConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use log::info;
use ndarray::{Array2, Array3, Axis};
use ort::{inputs, CPUExecutionProvider, GraphOptimizationLevel, Session};
use tokenizers::{Encoding, PaddingParams, Tokenizer, TruncationParams};
use tracing::error;

pub struct OnnxBert {
    config: VectorizerConfig,
    meta: Meta,
    model: Session,
    tokenizer: Tokenizer,
}

impl OnnxBert {
    pub fn new(
        graph: PathBuf,
        tokenizer: PathBuf,
        vectorizer_config: VectorizerConfig,
    ) -> OnnxBert {
        ort::init()
            .with_name("ort")
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit()
            .unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer).unwrap();
        let model = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .commit_from_file(graph)
            .unwrap();
        OnnxBert {
            config: vectorizer_config,
            meta: Meta {
                model_type: "onnx".to_string(),
            },
            model,
            tokenizer,
        }
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
            .with_truncation(Some({
                TruncationParams {
                    strategy: tokenizers::TruncationStrategy::LongestFirst,
                    ..Default::default()
                }
            }))
            .unwrap()
            .encode_batch(texts.clone(), true)
        {
            Ok(encodings) => Ok(encodings),
            Err(e) => {
                error!("error tokenizing batch: {:?}", e);
                return Err("error tokenizing batch");
            }
        }
    }
    fn get_inputs_from_encodings(
        &self,
        encodings: Vec<Encoding>,
    ) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
        let (attention_masks, input_ids, token_type_ids) = encodings.iter().fold(
            (
                ((0, 0), Vec::new()),
                ((0, 0), Vec::new()),
                ((0, 0), Vec::new()),
            ),
            |(mut attention_masks, mut input_ids, mut token_type_ids), encoding| {
                let mut am = encoding
                    .get_attention_mask()
                    .iter()
                    .map(|i| *i as i64)
                    .collect::<Vec<i64>>();
                let am_len = am.len();
                let mut iids = encoding
                    .get_ids()
                    .iter()
                    .map(|i| *i as i64)
                    .collect::<Vec<i64>>();
                let iids_len = iids.len();
                let mut ttids = encoding
                    .get_type_ids()
                    .iter()
                    .map(|i| *i as i64)
                    .collect::<Vec<i64>>();
                let ttids_len = ttids.len();

                attention_masks.1.append(&mut am);
                input_ids.1.append(&mut iids);
                token_type_ids.1.append(&mut ttids);

                attention_masks.0 = ((attention_masks.0).0 + 1 as usize, am_len);
                input_ids.0 = ((input_ids.0).0 + 1 as usize, iids_len);
                token_type_ids.0 = ((token_type_ids.0).0 + 1 as usize, ttids_len);

                (attention_masks, input_ids, token_type_ids)
            },
        );
        (
            Array2::from_shape_vec(attention_masks.0, attention_masks.1).unwrap(),
            Array2::from_shape_vec(input_ids.0, input_ids.1).unwrap(),
            Array2::from_shape_vec(token_type_ids.0, token_type_ids.1).unwrap(),
        )
    }
    fn mean_pooling(&self, embeddings: Array3<f32>, attention_masks: Array2<i64>) -> Vec<Vec<f64>> {
        zip(
            embeddings.axis_iter(Axis(0)),
            attention_masks.axis_iter(Axis(0)),
        )
        .map(|(embedding, attention_mask)| {
            let binding = attention_mask.mapv(|elem| elem as f64).insert_axis(Axis(1));
            let expanded_mask = binding.broadcast(embedding.dim()).unwrap();
            self.l2_normalize(
                zip(
                    (embedding.mapv(|elem| elem as f64) * expanded_mask)
                        .sum_axis(Axis(0))
                        .to_vec(),
                    expanded_mask
                        .sum_axis(Axis(0))
                        .iter()
                        .map(|elem| f64::clamp(*elem, 1e-9, f64::INFINITY))
                        .collect::<Vec<_>>(),
                )
                .map(|(numerator, denominator)| numerator / denominator)
                .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>()
    }
    fn l2_normalize(&self, embeddings: Vec<f64>) -> Vec<f64> {
        let norm = f64::clamp(
            embeddings
                .iter()
                .map(|elem| elem.powi(2))
                .sum::<f64>()
                .sqrt(),
            1e-12,
            f64::INFINITY,
        );
        embeddings.iter().map(|elem| *elem / norm).collect()
    }
}

impl Vectorize for OnnxBert {
    fn get_meta(&self) -> Meta {
        self.meta.clone()
    }
    fn vectorize(
        &self,
        texts: Vec<String>,
        config: VectorInputConfig,
    ) -> Result<Vec<Vec<f64>>, &'static str> {
        let encodings = self.tokenize(texts)?;

        let input = self.get_inputs_from_encodings(encodings);
        let inputs = match inputs! {
            "attention_mask" => input.0.clone(),
            "input_ids" => input.1,
            "token_type_ids" => input.2
        } {
            Ok(inputs) => inputs,
            Err(e) => {
                error!("error creating input: {:?}", e);
                return Err("error creating input");
            }
        };

        let start = std::time::Instant::now();
        let outputs = match self.model.run(inputs) {
            Ok(outputs) => outputs,
            Err(e) => {
                error!("error running model: {:?}", e);
                return Err("error running model");
            }
        };
        let (dim, embeddings) = outputs["last_hidden_state"]
            .try_extract_raw_tensor::<f32>()
            .unwrap();
        info!("generating embeddings {:?} took {:?}", dim, start.elapsed());

        let out = Array3::from_shape_vec(
            (dim[0] as usize, dim[1] as usize, dim[2] as usize),
            embeddings.to_vec(),
        )
        .unwrap();

        Ok(self.mean_pooling(out, input.0))
    }
}
