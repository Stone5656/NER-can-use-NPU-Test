# ner_openvino/download_model/builders/npu.py
from __future__ import annotations
from pathlib import Path

from optimum.intel.openvino import OVModelForTokenClassification

from ner_openvino.download_model.builders.base import OpenVINOModelBuilder
from ner_openvino.download_model.device_enum import Device
from ner_openvino.download_model.shape_json import (
    get_shape_json_path,
    is_shape_same,
    write_shape_json,
)


class NPUOpenVINOBuilder(OpenVINOModelBuilder):
    """
    NPU 用：dynamic_shapes=False で IR を持ち、
    (batch_size, max_seq_len) に固定してから compile する。
    """

    def __init__(
        self,
        model_dir: Path,
        repo_id: str | None,
        device: Device,
        max_seq_len: int,
        batch_size: int,
        logger,
    ) -> None:
        super().__init__(model_dir=model_dir, repo_id=repo_id, device=device, logger=logger)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def _create_ov_model(self, *, export: bool) -> OVModelForTokenClassification:
        kwargs: dict = {"compile": False, "dynamic_shapes": False}
        if export:
            kwargs["export"] = True
        if self.device is not None:
            kwargs["device"] = self.device.value

        return OVModelForTokenClassification.from_pretrained(
            self.model_dir,
            **kwargs,
        )

    def _postprocess_model(self, ov_model: OVModelForTokenClassification) -> None:
        self.logger.info(
            "reshape を適用: batch=%d, seq_len=%d", self.batch_size, self.max_seq_len
        )
        shape_path = get_shape_json_path(self.model_dir)
        current_shape = {
            "batch_size": self.batch_size,
            "sequence_length": self.max_seq_len,
        }

        if self.ir_exported_newly:
            self.logger.info("新規 IR 生成のため shape.json を削除します")
            if shape_path.exists():
                shape_path.unlink()
            # reshape & compile を行い、shape.json を作り直す
            ov_model.reshape(
                batch_size=self.batch_size,
                sequence_length=self.max_seq_len,
            )
            ov_model.compile()
            write_shape_json(
                shape_path,
                {
                    "batch_size": self.batch_size,
                    "sequence_length": self.max_seq_len,
                },
            )
            return

        if is_shape_same(shape_path, current_shape):
            self.logger.info(
                "shape.json が現在の設定と一致しているため、compile() をスキップします"
            )
        else:
            self.logger.info(
                "shape が異なるため NPU 用に compile() を実行します: %s", current_shape
            )
            ov_model.reshape(**current_shape)
            ov_model.compile()
            write_shape_json(shape_path, current_shape)
            self.logger.info("shape.json を更新しました: %s", shape_path)
