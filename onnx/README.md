# ONNX Exporting

This directory contains a script to export and quantize models from HuggingFace to ONNX format using [optimum](https://github.com/huggingface/optimum).

## Using the script

```bash
python -m onnx.export
```

## Using optimum-cli

```bash
optimum-cli export onnx -m sentence-transformers/all-MiniLM-L6-v2 ./all-minilm-l6-v2
```