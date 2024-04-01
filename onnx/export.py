import os

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

path = "./onnx/models/flan-t5-base"
model = "google/flan-t5-base"

if not os.path.exists(path):
    os.makedirs(path)

ort_model = ORTModelForFeatureExtraction.from_pretrained(model, export=True)
tokenizer = AutoTokenizer.from_pretrained(model)

ort_model.save_pretrained(path)
tokenizer.save_pretrained(path)
