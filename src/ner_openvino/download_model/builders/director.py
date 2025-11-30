# ner_openvino/download_model/builders/director.py
from __future__ import annotations
import logging
from pathlib import Path

from ner_openvino.download_model.types import LoadedNER
from ner_openvino.download_model.device_enum import Device
from ner_openvino.download_model.builders.generic import GenericOpenVINOBuilder
from ner_openvino.download_model.builders.npu import NPUOpenVINOBuilder


class OpenVINODirector:
    """
    Device Enum に応じて適切な Builder を選択し、LoadedNER を組み立てる役。
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def build_ner(
        self,
        *,
        device: Device,
        model_dir: Path,
        repo_id: str | None = None,
        max_seq_len: int = 256,
        batch_size: int = 1,
    ) -> LoadedNER:
        if device is Device.NPU:
            builder = NPUOpenVINOBuilder(
                model_dir=model_dir,
                repo_id=repo_id,
                device=device,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                logger=self.logger,
            )
        else:
            builder = GenericOpenVINOBuilder(
                model_dir=model_dir,
                repo_id=repo_id,
                device=device,
                logger=self.logger,
            )

        return builder.build()
