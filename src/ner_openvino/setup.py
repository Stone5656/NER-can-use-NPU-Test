import os
from pathlib import Path
from ner_openvino.download_model.loader_intel_npu import load_npu_model_intel
from ner_openvino.download_model.loader_intel import load_model_intel
from ner_openvino.model_urls import get_model_urls

model_dir_gpu = Path(os.getenv("NER_SAVE_DIR", "models/facebook_roberta"))
model_dir_npu = Path(os.getenv("NER_SAVE_DIR_NPU", "models/facebook_roberta_npu"))
repo_id = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
max_sequence_length = 256
batch_size = 4

print(get_model_urls(repo_id=repo_id))

ner_models = load_model_intel(
    model_dir=model_dir_gpu,
)

ner_models = load_npu_model_intel(
    model_dir=model_dir_npu,
    repo_id=repo_id,
    max_seq_len=max_sequence_length,
    batch_size=batch_size
)
