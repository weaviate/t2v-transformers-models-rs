#!/usr/bin/env python3

import os
import sys
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from pathlib import Path


model_dir = os.getenv('MODEL_DIR')
model_name = os.getenv('MODEL_NAME', None)
if not model_name:
    print("Fatal: MODEL_NAME is required")
    print("Please set environment variable MODEL_NAME to a HuggingFace model name, see https://huggingface.co/models")
    sys.exit(1)

onnx_runtime = os.getenv('ONNX_RUNTIME')
if not onnx_runtime:
    onnx_runtime = "false"

onnx_cpu_arch = os.getenv('ONNX_CPU')
if not onnx_cpu_arch:
    onnx_cpu_arch = "arm64"

print(f"Downloading MODEL_NAME={model_name} with ONNX_RUNTIME={onnx_runtime} ONNX_CPU={onnx_cpu_arch}")

def download_onnx_model(model_name: str, model_dir: str) -> tuple[str, str]:
    # Download model and tokenizer
    onnx_path = Path(model_dir)
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_name, from_transformers=True)
    # Save model
    ort_model.save_pretrained(onnx_path)

    def save_to_file(filepath: str, content: str):
        with open(filepath, "w") as f:
            f.write(content)

    def save_quantization_info(arch: str):
        save_to_file(f"{model_dir}/onnx_quantization_info", arch)

    def quantization_config(onnx_cpu_arch: str):
        if onnx_cpu_arch.lower() == "avx512_vnni":
            print("Quantize Model for x86_64 (amd64) (avx512_vnni)")
            save_quantization_info("AVX-512")
            return AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        if onnx_cpu_arch.lower() == "arm64":
            print(f"Quantize Model for ARM64")
            save_quantization_info("ARM64")
            return AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
        # default is AMD64 (AVX2)
        print(f"Quantize Model for x86_64 (amd64) (AVX2)")
        save_quantization_info("amd64 (AVX2)")
        return AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

    # Quantize the model / convert to ONNX
    qconfig = quantization_config(onnx_cpu_arch)
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    # Apply dynamic quantization on the model
    quantizer.quantize(save_dir=onnx_path, quantization_config=qconfig)
    # Remove model.onnx file, leave only model_quantized.onnx
    if os.path.isfile(f"{model_dir}/model.onnx"):
        os.remove(f"{model_dir}/model.onnx")
    # Save information about ONNX runtime
    save_to_file(f"{model_dir}/onnx_runtime", onnx_runtime)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(onnx_path)
    return model_dir

def download_model(model_name: str, model_dir: str) -> str:
    print(f"Downloading model {model_name} from huggingface model hub")
    config = AutoConfig.from_pretrained(model_name)

    print(f"Using class {config.architectures[0]} to load model weights")
    mod = __import__('transformers', fromlist=[config.architectures[0]])
    try:
        klass_architecture = getattr(mod, config.architectures[0])
        model = klass_architecture.from_pretrained(model_name)
    except AttributeError:
        print(f"{config.architectures[0]} not found in transformers, fallback to AutoModel")
        model = AutoModel.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir

if onnx_runtime == "true":
    download_onnx_model(model_name, model_dir)
else:
    download_model(model_name, model_dir)