"""
NERモデルの出力（トークンクラス分類のロジット）を人間可読なエンティティへ復号するユーティリティ。

本モジュールは、ロジット→確率化（softmax）→BIO解釈→スパン結合→
同ラベルの隣接スパンのマージ、という処理を関数ごとに分離して提供します。
ドキュメンテーションは PEP 257 に従い、要約行・空行・詳細説明の順で記述しています。
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np


def compute_softmax(prediction_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ロジットからsoftmax確率・最尤ラベルID・スコアを計算する。

    入力ロジットは形状 `(1, sequence_length, num_labels)` を想定します。
    最終次元に対してsoftmaxを適用し、各トークン位置ごとの
    「最尤ラベルID」と「その確率（スコア）」を返します。

    Args:
        prediction_logits (numpy.ndarray):
            形状 `(1, L, C)` のロジット。Lは系列長、Cはラベル数。

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - `predicted_label_ids`: 形状 `(L,)` の最尤ラベルID配列
            - `predicted_scores`   : 形状 `(L,)` の最尤ラベル確率（スコア）

    Notes:
        本関数はバッチ次元が1であることを前提とし、内部で squeeze します。
        複数バッチを扱う場合は、事前に対象バッチを選択してから本関数へ渡してください。
    """
    # 数値安定性のために最大値を引いてから指数化
    logits = prediction_logits.astype(np.float32)
    logits_max = logits.max(axis=-1, keepdims=True)
    shifted_logits = logits - logits_max

    exp_logits = np.exp(shifted_logits)
    sum_exp = exp_logits.sum(axis=-1, keepdims=True)
    probabilities = exp_logits / sum_exp  # (1, L, C)

    # バッチ次元（=1）を除去
    probabilities = probabilities[0]      # (L, C)

    predicted_label_ids = probabilities.argmax(axis=-1)  # (L,)
    predicted_scores = probabilities.max(axis=-1)        # (L,)
    return predicted_label_ids, predicted_scores


def parse_label_prefix(label: str) -> tuple[str, str]:
    """BIO接頭辞とエンティティ種別にラベルを分解する。

    `B-PER` のようなBIO表記を想定し、`-` を境に prefix と type に分割します。
    `-` を含まない場合は、タイプ全体に対して prefix=`"B"` がデフォルトで付く想定とします。

    Args:
        label (str): 例 `"B-PER"`, `"I-ORG"`, `"O"` など。

    Returns:
        tuple[str, str]: `(prefix, entity_type)`, 例：`("B", "PER")`。
                         `"O"` は `( "B", "O")` として扱います。
    """
    if "-" in label:
        prefix, entity_type = label.split("-", 1)
    else:
        prefix, entity_type = "B", label
    return prefix, entity_type


def output_ner_score(current_entity_state: dict[str, Any], original_text: str) -> Optional[dict[str, Any]]:
    """現在構築中のエンティティ状態を最終化して辞書へ変換する。

    BIO走査中に連続トークンでエンティティを拡張していき、
    切れ目（`O`やタイプ変更・特殊トークン等）で確定させる際に使用します。

    Args:
        current_entity_state (dict):
            `{"label", "start", "end", "scores"}` を持つ作業用の状態。
        original_text (str):
            元のテキスト（スパン抽出に使用）。

    Returns:
        Optional[dict]:
            有効なエンティティがあれば `{"text","label","score","start","end"}` を返し、
            なければ `None` を返します。
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


def validate_blank_gap(gap_substring: str) -> bool:
    """2スパン間のギャップ文字列が空白のみかを判定する。

    連結可能性の判定で、空白（スペース・改行等）のみであれば
    「隣接」と同等に扱って連結を許可します。

    Args:
        gap_substring (str): 2つのスパンの間のテキスト。

    Returns:
        bool: 空白のみなら True、そうでなければ False。
    """
    return gap_substring.strip() == ""


def is_continuous_label(previous_entity: dict[str, Any], current_entity: dict[str, Any], original_text: str) -> bool:
    """同一ラベルかつ隣接（重なり・接触・空白のみ）の場合に連結可能とみなす。

    Args:
        previous_entity (dict): 直前に確定したエンティティ。
        current_entity  (dict): 判定対象の次エンティティ。
        original_text   (str): ギャップ確認のための元テキスト。

    Returns:
        bool: 連結可能なら True。
    """
    # ラベルが異なるなら即不可
    if previous_entity["label"] != current_entity["label"]:
        return False

    # 2スパン間のギャップ文字列を取得
    gap_text = original_text[previous_entity["end"]: current_entity["start"]]

    # 「重なり/接触」または「ギャップが空白のみ」なら隣接扱い
    contiguous = (current_entity["start"] <= previous_entity["end"]) or validate_blank_gap(gap_text)
    return contiguous


def merge_adjacent_entities(entity_list: list[dict[str, Any]], original_text: str) -> list[dict[str, Any]]:
    """同一ラベルで隣接（空白含む）する連続スパンを結合し、1つのエンティティに統合する。

    スパン断片が連続して出るケース（日本語のサブワード分割など）を想定し、
    隣接・空白のみを挟む場合は1つのスパンにまとめます。
    スコアはスパン長に基づく長さ加重平均で再計算します。

    Args:
        entity_list   (list[dict]): 結合前のエンティティ配列。
        original_text (str)       : 元テキスト。

    Returns:
        list[dict]: 結合後のエンティティ配列。
    """
    if not entity_list:
        return entity_list

    merged_entities: list[dict[str, Any]] = [entity_list[0]]

    for current_entity in entity_list[1:]:
        last_entity = merged_entities[-1]

        if is_continuous_label(last_entity, current_entity, original_text):
            # 長さ（0割防止で最低1）を用いた加重平均
            prev_length = max(1, last_entity["end"] - last_entity["start"])
            curr_length = max(1, current_entity["end"] - current_entity["start"])
            total_length = prev_length + curr_length
            weighted_score = (last_entity["score"] * prev_length + current_entity["score"] * curr_length) / total_length

            merged_entities[-1] = {
                "text": original_text[last_entity["start"]: max(last_entity["end"], current_entity["end"])],
                "label": last_entity["label"],
                "score": float(weighted_score),
                "start": last_entity["start"],
                "end": max(last_entity["end"], current_entity["end"]),
            }
        else:
            merged_entities.append(current_entity)

    return merged_entities


def decode_ner_outputs(text: str, tokenizer, id2label: dict[int, str], logits: np.ndarray, max_sequence_length: int) -> list[dict[str, Any]]:
    """モデルロジット `(1, L, C)` からエンティティ配列へ復号する（単文・バッチ1想定）。

    処理フロー:
        1) 固定長でトークナイズ（`offset_mapping` と `special_tokens_mask` を取得）
        2) softmaxにより各トークン位置のラベル確率を計算
        3) BIO解釈でエンティティスパンを構築
        4) 同ラベルで隣接するスパンをマージ

    Args:
        text (str):
            入力テキスト。
        tokenizer:
            Hugging Face系のトークナイザー。`return_offsets_mapping=True`,
            `return_special_tokens_mask=True` に対応していること（Fast推奨）。
        id2label (dict[int, str]):
            ラベルID→ラベル名のマップ。
        logits (numpy.ndarray):
            形状 `(1, L, C)` のロジット。
        max_sequence_length (int):
            トークナイズ時の最大長。**モデルのreshape時と一致必須**。

    Returns:
        list[dict]:
            `{"text","label","score","start","end"}` を要素にもつエンティティ配列。

    Raises:
        ValueError:
            `logits` が `(1, L, C)` でない場合。

    注意:
        NPUでは動的形状が使えないため、`max_sequence_length` と
        事前の `reshape(..., sequence_length=max_sequence_length)` を一致させてください。
    """
    # --- (1) トークナイズ（固定長・オフセット・特殊トークンマスク） -----------------
    tokenization = tokenizer(
        text,
        return_tensors=None,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets = tokenization["offset_mapping"]
    special_tokens_mask = tokenization["special_tokens_mask"]

    # ロジットの形状検証（単文・バッチ1を想定）
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape to be (1, L, C), got {logits.shape}")

    # --- (2) softmaxで各位置の最尤ラベルとスコアを算出 ----------------------------------
    predicted_ids, predicted_scores = compute_softmax(logits)

    # --- (3) BIOに基づくスパン構築 ------------------------------------------------------
    entities: list[dict[str, Any]] = []
    current_state: dict[str, Any] = {"label": None, "start": None, "end": None, "scores": []}

    for index, (span, special_flag) in enumerate(zip(offsets, special_tokens_mask)):
        start_offset, end_offset = span
        valid_offset = start_offset is not None and end_offset is not None and end_offset > start_offset

        # 特殊トークンや無効オフセットは区切り扱い（現在のスパンを確定してリセット）
        if special_flag == 1 or not valid_offset:
            entity = output_ner_score(current_state, text)
            if entity:
                entities.append(entity)
            current_state = {"label": None, "start": None, "end": None, "scores": []}
            continue

        label_id = int(predicted_ids[index])
        predicted_label = id2label.get(label_id, "O")
        score = float(predicted_scores[index])

        prefix, entity_type = parse_label_prefix(predicted_label)

        if entity_type == "O":
            entity = output_ner_score(current_state, text)
            if entity:
                entities.append(entity)
            current_state = {"label": None, "start": None, "end": None, "scores": []}
            continue

        # 新規開始（B-）またはタイプ変更時はフラッシュして新スパン開始
        if prefix == "B" or (current_state["label"] is not None and current_state["label"] != entity_type):
            entity = output_ner_score(current_state, text)
            if entity:
                entities.append(entity)
            current_state = {"label": entity_type, "start": start_offset, "end": end_offset, "scores": [score]}
        else:
            # 継続（I- 同タイプ）。I-から始まる例外もB扱いで開始。
            if current_state["label"] is None:
                current_state = {"label": entity_type, "start": start_offset, "end": end_offset, "scores": [score]}
            else:
                current_state["end"] = end_offset
                current_state["scores"].append(score)

    # 末尾に残ったスパンを確定
    final_entity = output_ner_score(current_state, text)
    if final_entity:
        entities.append(final_entity)

    # --- (4) 同ラベル・隣接スパンのマージ ------------------------------------------------
    return merge_adjacent_entities(entities, text)
