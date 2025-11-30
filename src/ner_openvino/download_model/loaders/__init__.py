# ner_openvino/download_model/loaders/__init__.py
from .intel import load_model_intel
from .intel_npu import load_npu_model_intel

__all__ = ["load_model_intel", "load_npu_model_intel"]
