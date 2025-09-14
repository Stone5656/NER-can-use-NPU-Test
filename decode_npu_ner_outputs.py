"""
NERモデルの出力（トークンクラス分類のロジット）を人間可読なエンティティへ復号するユーティリティ（バッチ対応版）。

本モジュールは、ロジット→確率化（softmax）→BIO解釈→スパン結合→
同ラベルの隣接スパンのマージ、という処理を関数ごとに分離して提供します。
単文向けAPI（decode_ner_outputs）に加えて、複数テキスト＆ロジットを同時に扱う
decode_ner_outputs_batch を提供します。
"""

from __future__ import annotations
from typing import Any, Optional, Iterable, List, Tuple, Dict
import numpy as np


# =========================================================
# Softmax / ラベル決定
# =========================================================
def compute_softmax(prediction_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """【後方互換】ロジットからsoftmax確率・最尤ラベルID・スコア（単バッチ）を計算する。

    期待形状: (1, L, C)。内部で batch 次元を squeeze して (L, C) として扱う。
    """
    if prediction_logits.ndim != 3 or prediction_logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape to be (1, L, C), got {prediction_logits.shape}")
    logits = prediction_logits.astype(np.float32)
    logits_max = logits.max(axis=-1, keepdims=True)
    shifted = logits - logits_max
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)  # (1, L, C)
    probs = probs[0]  # (L, C)
    predicted_label_ids = probs.argmax(axis=-1)  # (L,)
    predicted_scores = probs.max(axis=-1)        # (L,)
    return predicted_label_ids, predicted_scores


def compute_softmax_batch(prediction_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ロジットからsoftmax確率・最尤ラベルID・スコア（バッチ）を計算する。

    期待形状: (B, L, C)。返り値は
      - predicted_label_ids: (B, L)
      - predicted_scores   : (B, L)
    """
    if prediction_logits.ndim != 3:
        raise ValueError(f"Expected logits shape to be (B, L, C), got {prediction_logits.shape}")
    logits = prediction_logits.astype(np.float32)
    logits_max = logits.max(axis=-1, keepdims=True)        # (B, L, 1)
    shifted = logits - logits_max
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)  # (B, L, C)
    predicted_label_ids = probs.argmax(axis=-1)  # (B, L)
    predicted_scores   = probs.max(axis=-1)      # (B, L)
    return predicted_label_ids, predicted_scores


# =========================================================
# BIOラベルユーティリティ
# =========================================================
def parse_label_prefix(label: str) -> tuple[str, str]:
    """BIO接頭辞とエンティティ種別にラベルを分解する。"""
    if "-" in label:
        prefix, entity_type = label.split("-", 1)
    else:
        prefix, entity_type = "B", label
    return prefix, entity_type


def output_ner_score(current_entity_state: dict[str, Any], original_text: str) -> Optional[dict[str, Any]]:
    """現在構築中のエンティティ状態を最終化して辞書へ変換する。"""
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
    """2スパン間のギャップ文字列が空白のみかを判定する。"""
    return gap_substring.strip() == ""


def is_continuous_label(previous_entity: dict[str, Any], current_entity: dict[str, Any], original_text: str) -> bool:
    """同一ラベルかつ隣接（重なり・接触・空白のみ）の場合に連結可能とみなす。"""
    if previous_entity["label"] != current_entity["label"]:
        return False
    gap_text = original_text[previous_entity["end"]: current_entity["start"]]
    contiguous = (current_entity["start"] <= previous_entity["end"]) or validate_blank_gap(gap_text)
    return contiguous


def merge_adjacent_entities(entity_list: list[dict[str, Any]], original_text: str) -> list[dict[str, Any]]:
    """同一ラベルで隣接（空白含む）する連続スパンを結合し、1つのエンティティに統合する。"""
    if not entity_list:
        return entity_list

    merged: list[dict[str, Any]] = [entity_list[0]]
    for current in entity_list[1:]:
        last = merged[-1]
        if is_continuous_label(last, current, original_text):
            prev_length = max(1, last["end"] - last["start"])
            curr_length = max(1, current["end"] - current["start"])
            total_length = prev_length + curr_length
            weighted_score = (last["score"] * prev_length + current["score"] * curr_length) / total_length

            merged[-1] = {
                "text": original_text[last["start"]: max(last["end"], current["end"])],
                "label": last["label"],
                "score": float(weighted_score),
                "start": last["start"],
                "end": max(last["end"], current["end"]),
            }
        else:
            merged.append(current)
    return merged


# =========================================================
# 復号（単文 / バッチ）
# =========================================================
def _decode_single(
    text: str,
    tokenizer,
    id2label: Dict[int, str],
    predicted_ids_1d: np.ndarray,
    predicted_scores_1d: np.ndarray,
    offsets: Iterable[Tuple[int, int]],
    special_tokens_mask: Iterable[int],
) -> list[dict[str, Any]]:
    """1サンプルぶん（L次元）のBIO走査→スパン構築→隣接マージ。"""
    entities: list[dict[str, Any]] = []
    state: dict[str, Any] = {"label": None, "start": None, "end": None, "scores": []}

    for index, (span, special_flag) in enumerate(zip(offsets, special_tokens_mask)):
        start_offset, end_offset = span
        valid_offset = start_offset is not None and end_offset is not None and end_offset > start_offset

        # 特殊 or 無効オフセット → 区切り
        if special_flag == 1 or not valid_offset:
            e = output_ner_score(state, text)
            if e:
                entities.append(e)
            state = {"label": None, "start": None, "end": None, "scores": []}
            continue

        label_id = int(predicted_ids_1d[index])
        predicted_label = id2label.get(label_id, "O")
        score = float(predicted_scores_1d[index])

        prefix, entity_type = parse_label_prefix(predicted_label)

        if entity_type == "O":
            e = output_ner_score(state, text)
            if e:
                entities.append(e)
            state = {"label": None, "start": None, "end": None, "scores": []}
            continue

        # 新規開始（B-）またはタイプ変更
        if prefix == "B" or (state["label"] is not None and state["label"] != entity_type):
            e = output_ner_score(state, text)
            if e:
                entities.append(e)
            state = {"label": entity_type, "start": start_offset, "end": end_offset, "scores": [score]}
        else:
            # 継続（I- 同タイプ）/ I-から開始もB扱いで開始
            if state["label"] is None:
                state = {"label": entity_type, "start": start_offset, "end": end_offset, "scores": [score]}
            else:
                state["end"] = end_offset
                state["scores"].append(score)

    # 末尾 flush
    last = output_ner_score(state, text)
    if last:
        entities.append(last)

    return merge_adjacent_entities(entities, text)


def decode_ner_outputs(
    text: str,
    tokenizer,
    id2label: Dict[int, str],
    logits: np.ndarray,
    max_sequence_length: int,
) -> list[dict[str, Any]]:
    """【後方互換】モデルロジット (1, L, C) からエンティティ配列へ復号（単文・バッチ1）。"""
    # トークナイズ（固定長）
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
    special = tokenization["special_tokens_mask"]

    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape to be (1, L, C), got {logits.shape}")

    pred_ids_1d, pred_scores_1d = compute_softmax(logits)
    return _decode_single(
        text=text,
        tokenizer=tokenizer,
        id2label=id2label,
        predicted_ids_1d=pred_ids_1d,
        predicted_scores_1d=pred_scores_1d,
        offsets=offsets,
        special_tokens_mask=special,
    )


def decode_ner_outputs_batch(
    texts: List[str],
    tokenizer,
    id2label: Dict[int, str],
    logits: np.ndarray,
    max_sequence_length: int,
) -> List[List[dict[str, Any]]]:
    """モデルロジット (B, L, C) からエンティティ配列（サンプルごと）へ復号（バッチ対応）。

    Args:
        texts (list[str]): バッチ内の原文テキスト列。長さ B。
        tokenizer:  Hugging Face Fast系推奨。offset_mapping と special_tokens_mask を返せること。
        id2label (dict[int, str]): ラベルID→ラベル名マップ。
        logits (np.ndarray): 形状 (B, L, C) のロジット。
        max_sequence_length (int): トークナイズ長（NPU等で固定長と一致させる）。

    Returns:
        list[list[dict]]: バッチ各要素に対応したエンティティ配列。
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits shape to be (B, L, C), got {logits.shape}")
    B, L, C = logits.shape
    if len(texts) != B:
        raise ValueError(f"len(texts) ({len(texts)}) must equal batch size B ({B}).")

    # 一括トークナイズ（固定長でパディング）
    tokenization = tokenizer(
        texts,
        return_tensors=None,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets_batch = tokenization["offset_mapping"]            # List[List[Tuple[int,int]]], 各長さ L
    special_batch = tokenization["special_tokens_mask"]       # List[List[int]], 各長さ L

    # softmax + argmax をバッチで計算
    pred_ids, pred_scores = compute_softmax_batch(logits)     # (B, L), (B, L)

    results: List[List[dict[str, Any]]] = []
    for b in range(B):
        # 念のため長さ整合を確認（ tokenizer 側 L と logits 側 L ）
        if len(offsets_batch[b]) != L or len(special_batch[b]) != L:
            raise ValueError(
                f"Tokenized length (len={len(offsets_batch[b])}) "
                f"must match logits length L ({L}) for sample {b}."
            )

        ents = _decode_single(
            text=texts[b],
            tokenizer=tokenizer,
            id2label=id2label,
            predicted_ids_1d=pred_ids[b],
            predicted_scores_1d=pred_scores[b],
            offsets=offsets_batch[b],
            special_tokens_mask=special_batch[b],
        )
        results.append(ents)

    return results


# 明示的に公開するAPI
__all__ = [
    "compute_softmax",
    "compute_softmax_batch",
    "parse_label_prefix",
    "output_ner_score",
    "validate_blank_gap",
    "is_continuous_label",
    "merge_adjacent_entities",
    "decode_ner_outputs",
    "decode_ner_outputs_batch",
]
