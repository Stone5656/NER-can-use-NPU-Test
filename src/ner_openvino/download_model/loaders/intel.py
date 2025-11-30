# ner_openvino/download_model/loaders/intel.py
from __future__ import annotations
import logging
from pathlib import Path

from ner_openvino.download_model.types import LoadedNER
from ner_openvino.download_model.device_enum import Device
from ner_openvino.download_model.builders import OpenVINODirector
from ner_openvino.utils.logger_utils.logger_injector import with_logger


@with_logger("NER-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def load_model_intel(
    model_dir: Path,
    *,
    repo_id: str | None = None,
    device: Device | str = Device.GPU,
    logger: logging.Logger,
) -> LoadedNER:
    """
    AUTO/CPU/GPU 用のエントリポイント。
    """
    if isinstance(device, str):
        device = Device.from_str(device)

    director = OpenVINODirector(logger=logger)
    return director.build_ner(
        device=device,
        model_dir=model_dir,
        repo_id=repo_id,
    )
