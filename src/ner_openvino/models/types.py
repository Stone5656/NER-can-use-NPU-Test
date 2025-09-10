"""
models/types.py
NER 関連の型定義。
"""
from __future__ import annotations
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, PreTrainedModel

@dataclass
class LoadedNER:
    model_dir: str
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    id2label: dict[int, str]
    label2id: dict[str, int]
