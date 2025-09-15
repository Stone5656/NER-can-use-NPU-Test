"""bio_interpretation.py

BIO 形式のラベルを扱うための小さなユーティリティ。

- `parse_label_prefix(label)`:
    "B-PER" のような BIO ラベルを、先頭の記号 (B/I/O など) と
    種類名 (PER/ORG など) に分けます。"-" が無い場合は "B" を付けた扱いにします。
    例: "O" -> ("B", "O")

- `output_ner_score(state, original_text)`:
    BIO を左から読み進めるとき、**Inside (I-*) が続く限り** 同じ単語のまとまりとして
    伸ばし、"O" に出会ったり種類が変わったところで確定させます。
    確定時に、対象の元テキストの切り出し、平均スコア、開始/終了位置などを
    辞書にして返します。何も確定できない場合は None を返します。

用語はできるだけ平易にしています（「単語」「まとまり」など）。
"""

from __future__ import annotations
import numpy as np
from ner_openvino.utils.logger_utils.logger_utils import LoggerFactoryImpl

logger = LoggerFactoryImpl("NER-OpenVINO-APP", log_file="logs/app.log")


# BIOラベル付きの文字をラベルと文字に分解
def parse_label_prefix(label: str) -> tuple[str, str]:
    """BIO ラベルを「接頭記号」と「種類名」に分ける。

    想定する入力:
        - "B-PER", "I-ORG" のように "-" で区切られた表記
        - "O" のように "-" を含まない表記

    ふるまい:
        - "-" を含む場合: "-" の手前を prefix、後ろを種類名として返す
        - "-" を含まない場合: prefix は "B"、種類名はそのまま扱う
          例: "O" → ("B", "O")

    Args:
        label (str): 例 "B-PER", "I-ORG", "O" など。

    Returns:
        tuple[str, str]: (prefix, 種類名) の組。例: ("B", "PER")
    """
    if "-" in label:
        prefix, entity_type = label.split("-", 1)
    else:
        prefix, entity_type = "B", label
    return prefix, entity_type


# 単語情報を登録
def output_ner_score(current_entity_state: dict[str, object], original_text: str) -> dict[str, object] | None:
    """現在まとまりつつある単語情報を「確定」して辞書にして返す。

    使いどころ:
        BIO を順に読んでいく処理の中で、
        **Inside (I-*) が続く限り** 同じ単語のまとまりとして伸ばし、
        "O" に出会ったり種類が変わったタイミングで、この関数で確定させます。

    入力:
        current_entity_state (dict):
            次のキーを持つ想定:
              - "label": 種類名 (例: "PER", "LOC" など)
              - "start": 元テキスト中の開始位置 (int)
              - "end"  : 元テキスト中の終了位置 (int, Python のスライス終端と同じ扱い)
              - "scores": 途中で集めたスコアのリスト (任意)

        original_text (str):
            元の文章。確定時に [start:end] で文字列を切り出します。

    返り値:
        dict[str, object] | None:
            確定できる内容があれば以下の形で返します。無ければ None。
            {
                "text":  切り出した文字列,
                "label": 種類名,
                "score": 平均スコア (scores が空なら 0.0),
                "start": 開始位置,
                "end":   終了位置,
            }
    """
    entity_label = current_entity_state.get("label")
    if entity_label is None:
        return None

    start_index = current_entity_state["start"]
    end_index = current_entity_state["end"]
    scores = current_entity_state.get("scores", [])

    entity_text = original_text[start_index:end_index]
    mean_score = float(np.mean(scores)) if scores else 0.0

    return {
        "text": entity_text,
        "label": entity_label,
        "score": mean_score,
        "start": start_index,
        "end": end_index,
    }
