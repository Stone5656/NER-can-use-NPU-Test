"""
models/config.py
allow/ignore パターンの読み込みと、既定モデルIDの定義。
"""
from __future__ import annotations
from pathlib import Path
from ner_openvino.utils.text_utils.read_line import clean_lines

DEFAULT_MODEL_REPO = "tsmatz/xlm-roberta-ner-japanese"

def _resolve_models_dir() -> Path:
    # このファイルの隣にある ner_allow.txt / ner_ignore.txt を読む
    return Path(__file__).resolve().parent

def load_allow_patterns() -> list[str]:
    return clean_lines(str(_resolve_models_dir() / "ner_allow.txt"))

def load_ignore_patterns() -> list[str]:
    return clean_lines(str(_resolve_models_dir() / "ner_ignore.txt"))
