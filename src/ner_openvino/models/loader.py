"""
models/loader.py
Tokenizer/Model のロードとラベルマップ取得。
"""
from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForTokenClassification
from ner_openvino.utils.logger_utils.logger_utils import LoggerFactoryImpl

from ner_openvino.models.downloader import download_model_snapshot
from ner_openvino.models.types import LoadedNER

logger = LoggerFactoryImpl("NER-OpenVINO-APP", log_file="logs/app.log")

def load_ner_model(
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
    device_map: str | None = None,
) -> LoadedNER:
    model_dir = download_model_snapshot(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token
    )

    logger.info("Tokenizer をロードしています（use_fast=True）")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    logger.info("NER モデルをロードしています")
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        torch_dtype=None,          # Transformers 既定
        low_cpu_mem_usage=True,
        device_map=device_map,     # None=CPU / "auto"=自動割当
    )

    config = model.config
    # config.id2label は {str:int} or {int:str} 両パターンがあり得るのでケア
    raw_id2label = getattr(config, "id2label", {})
    id2label: dict[int, str] = {int(k): v for k, v in raw_id2label.items()} if raw_id2label else {}
    label2id: dict[str, int] = getattr(config, "label2id", {})

    logger.info(f"ラベル数: {getattr(config, 'num_labels', len(id2label))}")
    logger.debug(f"id2label: {id2label}")
    logger.debug(f"label2id: {label2id}")

    # offset_mapping が FastTokenizer で有効か確認
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
