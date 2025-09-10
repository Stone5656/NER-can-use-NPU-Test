from __future__ import annotations
from pathlib import Path

from optimum.intel.openvino import OVModelForTokenClassification
from transformers import AutoTokenizer
from ner_openvino.utils.logger_utils.logger_utils import LoggerFactoryImpl
from ner_openvino.download_model.types import LoadedNER
from src.ner_openvino.download_model.downloader import download_model_snapshot

logger = LoggerFactoryImpl("NER-OpenVINO-APP", log_file="logs/app.log")

def load_ner_model_intel(model_dir: Path) -> LoadedNER:
    """
    既存の model_dir から OpenVINO IR をロード（なければDL→IR出力→保存）。
    """
    logger.info("NER モデルをロード/変換します")

    # 1) 事前にスナップショットがなければ取得
    if not model_dir.exists() or not any(model_dir.glob("*.bin")):
        model_dir.mkdir(parents=True, exist_ok=True)
        # 例: リポジトリIDは downloader 側のデフォルトを使用
        download_model_snapshot(save_dir=model_dir)
        # 2) ここで必ず IR へエクスポート（オンザフライ変換）
        ov_model = OVModelForTokenClassification.from_pretrained(
            model_dir,
            export=True,           # ← これが IR 変換のトリガー
            compile=False,
        )
        # 3) IR をディスク保存（openvino_model.xml / .bin が出力される）
        ov_model.save_pretrained(model_dir)
    else:
        # 既に IR がある（.xml/.bin）のでそのまま読み込み
        ov_model = OVModelForTokenClassification.from_pretrained(
            model_dir,
            compile=False,
        )

    # 必要に応じてデバイスコンパイル（AUTO/CPU/GPU/NPU）
    ov_model.compile()  # 既定は AUTO 相当の動作。明示指定するなら compile(device='AUTO')

    # Tokenizer
    logger.info("Tokenizer をロードしています（use_fast=True）")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    config = ov_model.config
    raw_id2label = getattr(config, "id2label", {})
    id2label: dict[int, str] = {int(k): v for k, v in raw_id2label.items()} if raw_id2label else {}
    label2id: dict[str, int] = getattr(config, "label2id", {})

    # offset_mapping の健全性チェック
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
