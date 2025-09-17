from __future__ import annotations
import logging
from pathlib import Path

from optimum.intel.openvino import OVModelForTokenClassification
from transformers import AutoTokenizer
from ner_openvino.download_model.types import LoadedNER
from ner_openvino.download_model.downloader import download_model_snapshot
from ner_openvino.utils.logger_utils.logger_injector import with_logger

@with_logger("NER-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_var="LOG_LEVEL")
def load_npu_model_intel(
    model_dir: Path,
    *,
    repo_id:str | None = None,
    device: str = "NPU",         # ← 既定でNPU
    max_seq_len: int = 256,      # ← NPU要件：固定長にそろえる
    batch_size: int = 1,         # ← NPU要件：バッチも固定（一般に1を推奨）
    logger: logging.Logger,
) -> LoadedNER:
    """
    NPU向け：既存の model_dir から OpenVINO IR をロード（なければDL→IR出力→保存）。
    その後、NPUが要求する静的形状（batch, seq_len）へreshapeしてcompile(device='NPU')する。
    """
    logger.info("NER モデルをロード/変換します（NPU/静的形状対応）")

    # 1) IRが無ければスナップショット取得→IRへエクスポート
    if not model_dir.exists() or not any(model_dir.glob("*.bin")):
        model_dir.mkdir(parents=True, exist_ok=True)
        # downloader側のデフォルトrepo_idで取得
        download_model_snapshot(
            repo_id=repo_id,
            save_dir=model_dir
        )

        # PyTorch重みからIRへオンザフライ変換
        ov_model = OVModelForTokenClassification.from_pretrained(
            model_dir,
            export=True,        # ← IR変換のトリガ
            compile=False,
            device=device,
            dynamic_shapes=False,
        )
        ov_model.save_pretrained(model_dir)  # openvino_model.xml / .bin を保存
        logger.info("IR へエクスポート済み: %s", model_dir)
    else:
        # 既存IRをそのままロード
        ov_model = OVModelForTokenClassification.from_pretrained(
            model_dir,
            compile=False,
            device=device,
            dynamic_shapes=False,
        )
        logger.info("既存IRをロード: %s", model_dir)

    # 2) NPU要件：静的形状に固定してからコンパイル
    logger.info("reshape を適用: batch=%d, seq_len=%d", batch_size, max_seq_len)
    ov_model.reshape(batch_size=batch_size, sequence_length=max_seq_len)  # ★重要

    # ★ compile() は「引数なし」で呼ぶ（device は上で設定済み）
    logger.info("NPU へコンパイルします")
    ov_model.compile()

    # 3) Tokenizer
    logger.info("Tokenizer をロード（use_fast=True）")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # NPUでは可変長は不可。後段の推論コードで必ず
    # padding='max_length', truncation=True, max_length=max_seq_len を指定すること。
    if getattr(tokenizer, "model_max_length", None) and tokenizer.model_max_length < max_seq_len:
        logger.warning(
            "tokenizer.model_max_length(%s) < max_seq_len(%s)。max_length指定で明示固定してください。",
            tokenizer.model_max_length, max_seq_len
        )

    # ラベルマップ回収
    config = ov_model.config
    raw_id2label = getattr(config, "id2label", {}) or {}
    id2label: dict[int, str] = {int(k): v for k, v in raw_id2label.items()}
    label2id: dict[str, int] = getattr(config, "label2id", {}) or {}

    # Fastトークナイザ健全性チェック（offset_mapping）
    test_text = "テスト用の短い文章です。"
    enc = tokenizer(test_text, return_offsets_mapping=True)
    if enc.get("offset_mapping", None) is None:
        logger.warning("offset_mapping が取得できません。Fast トークナイザか確認してください。")

    return LoadedNER(
        model_dir=model_dir,
        tokenizer=tokenizer,
        model=ov_model,
        id2label=id2label,
        label2id=label2id,
    )
