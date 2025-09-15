"""marge_space.py

テキスト中で同じラベルが付いた区間（まとまり）が、
空白や接触によって分かれてしまう場合に「ひとつにまとめる」ための補助関数を集めたモジュール。

提供する主な関数:
    - validate_blank_gap(gap_substring):
        2つの区間の間が空白だけかどうかを判定します。
        空白だけなら「つながっている」と見なせます。

    - is_continuous_label(previous_word, current_word, original_text):
        直前のまとまりと次のまとまりが同じラベルで、
        重なっている / くっついている / 空白だけを挟んでいる場合に
        「連続」として扱うかを判定します。

    - merge_adjacent_words(word_list, original_text):
        同じラベルで連続しているまとまりを結合し、
        ひとつのまとまりに統合します。
        サブワード分割などで細かく切れて出力された場合でも、
        実際の文意に沿ったまとまりとして整理できます。

利用イメージ:
    - NER処理などで出力された「部分的に切れたまとまり」のリストを渡すと、
      空白を挟んだ連続部分をまとめて1つにし、スコアは区間の長さで加重平均します。
"""

from __future__ import annotations
import numpy as np
from ner_openvino.utils.logger_utils.logger_utils import LoggerFactoryImpl

logger = LoggerFactoryImpl("NER-OpenVINO-APP", log_file="logs/app.log")


def validate_blank_gap(gap_substring: str) -> bool:
    """【一文】2つの区間のあいだが「空白だけ」かどうかを調べます。

    目的:
        まとまり同士をつなげてよいか考えるときに、
        あいだの文字が空白（スペース・改行など）だけなら
        「実質つながっている」と見なしやすくします。

    Args:
        gap_substring (str): 2つの区間の間にある文字列。

    Returns:
        bool: 空白だけなら True、そうでなければ False。

    Notes:
        - Pythonの `str.strip()` は、スペースやタブ、改行などの空白類を除去します。
          和文の全角空白（U+3000）も空白扱いになるため、一般的なケースはこれで十分です。
    """
    return gap_substring.strip() == ""


def is_continuous_label(previous_word: dict[str, object], current_word: dict[str, object], original_text: str) -> bool:
    """【一文】同じラベルで「重なる/くっつく/空白だけを挟む」関係なら、つなげられると判断します。

    目的:
        直前の区間と次の区間が同じラベルで、境界が重なっていたり、
        ぴったり接していたり、間が空白だけなら「連続している」と見なします。

    Args:
        previous_word (dict): 直前に確定したまとまり情報（"label","start","end" など）。
        current_word  (dict): 判定対象の次のまとまり情報。
        original_text (str): 元テキスト（区間間の文字を確認するために使います）。

    Returns:
        bool: 連結可能なら True、そうでなければ False。

    Notes:
        - まずラベルが同じかを確認し、違えば連結不可です。
        - 次に、`previous.end` と `current.start` の関係で
          「重なり/接触（`current.start <= previous.end`）」をチェックします。
        - それ以外でも、間のテキストが空白だけなら連結可とします（`validate_blank_gap` を使用）。
    """
    # ラベルが異なるなら即不可
    if previous_word["label"] != current_word["label"]:
        return False

    # 2つの区間の間にある文字列を取得
    gap_text = original_text[previous_word["end"]: current_word["start"]]

    # 「重なり/接触」または「間が空白のみ」なら隣接扱い
    contiguous = (current_word["start"] <= previous_word["end"]) or validate_blank_gap(gap_text)
    return contiguous


def merge_adjacent_entities(word_list: list[dict[str, object]], original_text: str) -> list[dict[str, object]]:
    """【一文】同じラベルで隣接（空白を含む）する区間をまとめて、ひとつのまとまりに統合します。

    目的:
        サブワード分割などで同じラベルのまとまりが細切れに出るとき、
        「重なる/くっつく/空白だけを挟む」関係なら1つのまとまりとして結合します。
        スコアは区間の長さに基づく長さ加重平均で再計算します。

    Args:
        word_list    (list[dict]): 結合前のまとまり配列（各要素は "text","label","score","start","end" を想定）。
        original_text (str)       : 元テキスト（結合後の "text" を切り出すために使います）。

    Returns:
        list[dict]: 結合後のまとまり配列（必要に応じて統合された結果）。

    Notes:
        - 空の入力はそのまま返します。
        - 長さ0のまとまりが来ても0割を避けるため、重みは最低1にしています。
        - 統合時の `end` は重なりを考慮して `max(end_i)` を採用します。
    """
    if not word_list:
        return word_list

    merged_words: list[dict[str, object]] = [word_list[0]]

    for current_word in word_list[1:]:
        last_word = merged_words[-1]

        if is_continuous_label(last_word, current_word, original_text):
            # 長さ（0割防止で最低1）を用いた加重平均
            prev_length = max(1, last_word["end"] - last_word["start"])
            curr_length = max(1, current_word["end"] - current_word["start"])
            total_length = prev_length + curr_length
            weighted_score = (last_word["score"] * prev_length + current_word["score"] * curr_length) / total_length

            merged_words[-1] = {
                "text": original_text[last_word["start"]: max(last_word["end"], current_word["end"])],
                "label": last_word["label"],
                "score": float(weighted_score),
                "start": last_word["start"],
                "end": max(last_word["end"], current_word["end"]),
            }
        else:
            merged_words.append(current_word)

    return merged_words
