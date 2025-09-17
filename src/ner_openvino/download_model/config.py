"""
models/config.py
allow/ignore パターンの読み込みと、既定モデルIDの定義。
"""
from __future__ import annotations
import os
from pathlib import Path
from ner_openvino.utils.text_utils.read_line import clean_lines

DEFAULT_MODEL_REPO = "tsmatz/xlm-roberta-ner-japanese"

def _resolve_models_dir() -> Path:
    # 環境変数 NER_PATTERN_DIR が設定されていればそちらを優先
    env_dir = os.getenv("NER_PATTERN_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    # そうでなければこのファイルの隣を使用
    return Path(__file__).resolve().parent

def load_allow_patterns() -> list[str]:
    folder_path = _resolve_models_dir()
    return clean_lines(str(folder_path / "ner_allow.txt"))

def load_ignore_patterns() -> list[str]:
    folder_path = _resolve_models_dir()
    return clean_lines(str(folder_path / "ner_ignore.txt"))
