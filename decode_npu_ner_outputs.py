"""
NERモデルの出力（トークンクラス分類のロジット）を人間可読なエンティティへ復号するユーティリティ。

本モジュールは、ロジット→確率化（softmax）→BIO解釈→スパン結合→
同ラベルの隣接スパンのマージ、という処理を関数ごとに分離して提供します。
ドキュメンテーションは PEP 257 に従い、要約行・空行・詳細説明の順で記述しています。
"""

from __future__ import annotations
import numpy as np
from ner_openvino.npu_ner.bio_interpretation import output_ner_score, parse_label_prefix
from ner_openvino.npu_ner.softmax import compute_softmax
from ner_openvino.npu_ner.marge_space import merge_adjacent_entities
from ner_openvino.utils.logger_utils.logger_utils import LoggerFactoryImpl

logger = LoggerFactoryImpl("NER-OpenVINO-APP", log_file="logs/app.log")


def decode_ner_outputs(text: str, tokenizer, id2label: dict[int, str], logits: np.ndarray, max_sequence_length: int) -> list[dict[str, object]]:
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
    entities: list[dict[str, object]] = []
    current_state: dict[str, object] = {"label": None, "start": None, "end": None, "scores": []}

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
