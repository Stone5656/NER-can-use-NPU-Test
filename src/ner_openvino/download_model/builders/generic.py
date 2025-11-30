# ner_openvino/download_model/builders/generic.py
from __future__ import annotations
from pathlib import Path

from optimum.intel.openvino import OVModelForTokenClassification

from ner_openvino.download_model.builders.base import OpenVINOModelBuilder
from ner_openvino.download_model.device_enum import Device


class GenericOpenVINOBuilder(OpenVINOModelBuilder):
    """
    AUTO / CPU / GPU 用。dynamic_shapes の指定は行わない。
    """

    def __init__(
        self,
        model_dir: Path,
        repo_id: str | None,
        device: Device,
        logger,
    ) -> None:
        super().__init__(model_dir=model_dir, repo_id=repo_id, device=device, logger=logger)

    def _create_ov_model(self, *, export: bool) -> OVModelForTokenClassification:
        kwargs: dict = {"compile": False}
        if export:
            kwargs["export"] = True
        if self.device is not None:
            kwargs["device"] = self.device.value

        return OVModelForTokenClassification.from_pretrained(
            self.model_dir,
            **kwargs,
        )

    def _postprocess_model(self, ov_model: OVModelForTokenClassification) -> None:
        self.logger.info("OpenVINO モデルを compile します (device=%s)", self.device.value)
        ov_model.compile()
