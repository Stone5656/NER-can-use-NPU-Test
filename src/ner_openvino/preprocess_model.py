import os
from pathlib import Path
from ner_openvino.download_model.loader_intel_npu import load_npu_model_intel
from ner_openvino.download_model.loader_intel import load_model_intel
from ner_openvino.setup import model_dir_gpu, model_dir_npu, repo_id, max_sequence_length, batch_size

ner_models = load_model_intel(
    model_dir=model_dir_gpu,
)

ner_models = load_npu_model_intel(
    model_dir=model_dir_npu,
    repo_id=repo_id,
    max_seq_len=max_sequence_length,
    batch_size=batch_size
)
