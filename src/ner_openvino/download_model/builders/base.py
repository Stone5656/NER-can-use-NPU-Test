# ner_openvino/download_model/builders/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from pathlib import Path

from optimum.intel.openvino import OVModelForTokenClassification
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ner_openvino.download_model.types import LoadedNER
from ner_openvino.download_model.downloader import download_model_snapshot
from ner_openvino.download_model.device_enum import Device


class OpenVINOModelBuilder(ABC):
    """
    IR 探索 → 変換 → OpenVINO モデル生成 → Tokenizer/ラベル構築までの共通処理。

    NPU などデバイス固有の処理は _postprocess_model で差し替える。
    """

    def __init__(
        self,
        model_dir: Path,
        repo_id: str | None,
        device: Device,
        logger: logging.Logger,
    ) -> None:
        self.model_dir = model_dir
        self.repo_id = repo_id
        self.device = device
        self.logger = logger
        self.ir_exported_newly = False

    # ==== テンプレートメソッド ====
    def build(self) -> LoadedNER:
        self.model_dir.mkdir(parents=True, exist_ok=True)

        has_ir, has_config, has_torch_weights = self._inspect_files()

        ov_model = self._load_or_convert(
            has_ir=has_ir,
            has_config=has_config,
            has_torch_weights=has_torch_weights,
        )

        # デバイス固有の後処理（NPU はここで reshape + shape.json 等）
        self._postprocess_model(ov_model)

        tokenizer, id2label, label2id = self._load_tokenizer_and_labels(ov_model)

        return LoadedNER(
            model_dir=self.model_dir,
            tokenizer=tokenizer,
            model=ov_model,
            id2label=id2label,
            label2id=label2id,
        )

    # ==== 共通実装部分 ====

    def _inspect_files(self) -> tuple[bool, bool, bool]:
        has_ir_xml = any(self.model_dir.glob("*.xml"))
        has_ir_bin = any(self.model_dir.glob("*.bin"))
        has_ir = has_ir_xml and has_ir_bin

        has_config = (self.model_dir / "config.json").exists()
        has_torch_weights = any(self.model_dir.glob("pytorch_model.bin")) or any(
            self.model_dir.glob("*.safetensors")
        )

        self.logger.info(
            "ファイル状況: has_ir=%s (xml=%s, bin=%s), has_config=%s, has_torch_weights=%s",
            has_ir,
            has_ir_xml,
            has_ir_bin,
            has_config,
            has_torch_weights,
        )
        return has_ir, has_config, has_torch_weights

    def _load_or_convert(
        self,
        *,
        has_ir: bool,
        has_config: bool,
        has_torch_weights: bool,
    ) -> OVModelForTokenClassification:
        """
        1. IR があればそれを使う
        2. IR は無いが config + weight があればローカル変換
        3. それも無理なら repo_id からダウンロードして変換
        """
        # 1. 既存 IR
        if has_ir:
            self.logger.info("既存の IR (*.xml / *.bin) を使用します: %s", self.model_dir)
            return self._create_ov_model(export=False)

        # 2. ローカル変換
        if has_config and has_torch_weights:
            self.logger.info(
                "IR はありませんが、ローカルに PyTorch 重みと config.json があるため、"
                "ローカル変換を試みます (export=True)"
            )
            ov_model = self._create_ov_model(export=True)
            self.ir_exported_newly = True
            ov_model.save_pretrained(self.model_dir)
            self.logger.info(
                "ローカルの PyTorch 重みから IR へエクスポートしました: %s", self.model_dir
            )
            return ov_model

        # 3. ダウンロード
        if not self.repo_id:
            self.logger.error(
                "IR もローカルの PyTorch 重みも見つからず、repo_id も指定されていません。"
                "ダウンロードによる取得ができないため処理を中止します。"
            )
            raise FileNotFoundError(
                f"IR nor local PyTorch weights not found in {self.model_dir}, "
                "and repo_id is None; cannot download model."
            )

        self.logger.info(
            "IR もローカル変換用の重みも無いため、repo_id=%s からダウンロードを行います",
            self.repo_id,
        )
        download_model_snapshot(
            repo_id=self.repo_id,
            save_dir=self.model_dir,
        )

        ov_model = self._create_ov_model(export=True)
        self.ir_exported_newly = True
        ov_model.save_pretrained(self.model_dir)
        self.logger.info(
            "ダウンロードした PyTorch 重みから IR へエクスポートしました: %s", self.model_dir
        )
        return ov_model

    def _load_tokenizer_and_labels(
        self,
        ov_model: OVModelForTokenClassification,
    ) -> tuple[PreTrainedTokenizerBase, dict[int, str], dict[str, int]]:
        self.logger.info("Tokenizer をロードしています（use_fast=True）")
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)

        config = ov_model.config
        raw_id2label = getattr(config, "id2label", {}) or {}
        id2label: dict[int, str] = {int(k): v for k, v in raw_id2label.items()}
        label2id: dict[str, int] = getattr(config, "label2id", {}) or {}

        test_text = "テスト用の短い文章です。"
        enc = tokenizer(test_text, return_offsets_mapping=True)
        if enc.get("offset_mapping", None) is None:
            self.logger.warning(
                "offset_mapping が取得できません。Fast トークナイザか確認してください。"
            )

        return tokenizer, id2label, label2id

    # ==== サブクラスが決める部分 ====

    @abstractmethod
    def _create_ov_model(self, *, export: bool) -> OVModelForTokenClassification:
        ...

    @abstractmethod
    def _postprocess_model(self, ov_model: OVModelForTokenClassification) -> None:
        ...
