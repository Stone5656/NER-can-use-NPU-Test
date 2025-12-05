import os
from pathlib import Path
from ner_openvino.download_model.config import DEFAULT_MODEL_REPO
from ner_openvino.model_urls import get_model_urls
import json

model_dir_gpu = Path(
    os.getenv("NER_SAVE_DIR", "models/facebook_roberta")
)
model_dir_npu = Path(
    os.getenv("NER_SAVE_DIR_NPU", "models/facebook_roberta_npu")
)

repo_id = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
max_sequence_length = 256
batch_size = 4

print(json.dumps({
    "type": "gpu",
    "savedir": str(model_dir_gpu),
    "urls": get_model_urls(repo_id=DEFAULT_MODEL_REPO)
}))
print(json.dumps({
    "type": "npu",
    "savedir": str(model_dir_npu),
    "urls": get_model_urls(repo_id=repo_id)
}))
