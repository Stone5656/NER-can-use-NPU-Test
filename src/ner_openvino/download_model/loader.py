"""
models/loader.py
Tokenizer/Model のロードとラベルマップ取得。
"""
from __future__ import annotations
from pathlib import Path

from src.ner_openvino.download_model.downloader import download_model_snapshot
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ner_openvino.utils.logger_utils.logger_utils import LoggerFactoryImpl
from ner_openvino.download_model.types import LoadedNER

logger = LoggerFactoryImpl("NER-OpenVINO-APP", log_file="logs/app.log")

def load_ner_model(
    model_dir: Path,                 # ← 必須化（既にダウンロード済みのディレクトリを指定）
    device_map: str | None = None,
) -> LoadedNER:
    """
    すでにローカルにある model_dir をロードする。
    ダウンロードは別途 download_model_snapshot を直接呼び出すこと。
    """

    logger.info("NER モデルをロードしています")
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
        model = download_model_snapshot(save_dir=model_dir)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            torch_dtype=None,          # Transformers 既定
            low_cpu_mem_usage=True,
            device_map=device_map,     # None=CPU / "auto"=自動割当
        )

    logger.info("Tokenizer をロードしています（use_fast=True）")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    config = model.config
    raw_id2label = getattr(config, "id2label", {})
    id2label: dict[int, str] = {int(k): v for k, v in raw_id2label.items()} if raw_id2label else {}
    label2id: dict[str, int] = getattr(config, "label2id", {})

    logger.info(f"ラベル数: {getattr(config, 'num_labels', len(id2label))}")
    logger.debug(f"id2label: {id2label}")
    logger.debug(f"label2id: {label2id}")

    # offset_mapping チェック
    test_text = "テスト用の短い文章です。"
    enc = tokenizer(test_text, return_offsets_mapping=True)
    if enc.get("offset_mapping", None) is None:
        logger.warning(
            "offset_mapping が取得できません（Fastトークナイザでない可能性）。"
            "後続のスパン復元に影響する恐れがあります。"
        )

    return LoadedNER(
        model_dir=model_dir,
        tokenizer=tokenizer,
        model=model,
        id2label=id2label,
        label2id=label2id,
    )
