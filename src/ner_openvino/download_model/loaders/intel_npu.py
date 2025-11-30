# ner_openvino/download_model/loaders/intel_npu.py
from __future__ import annotations
import logging
from pathlib import Path

from ner_openvino.download_model.types import LoadedNER
from ner_openvino.download_model.device_enum import Device
from ner_openvino.download_model.builders import OpenVINODirector
from ner_openvino.utils.logger_utils.logger_injector import with_logger


@with_logger("NER-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def load_npu_model_intel(
    model_dir: Path,
    *,
    repo_id: str | None = None,
    max_seq_len: int = 256,
    batch_size: int = 1,
    logger: logging.Logger,
) -> LoadedNER:
    """
    NPU 専用のエントリポイント。
    """
    director = OpenVINODirector(logger=logger)
    return director.build_ner(
        device=Device.NPU,
        model_dir=model_dir,
        repo_id=repo_id,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )
