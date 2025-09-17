from __future__ import annotations
import numpy as np
from collections.abc import Iterable

from ner_openvino.npu_ner.bio_interpretation import output_ner_score, parse_label_prefix
from ner_openvino.npu_ner.marge_space import merge_adjacent_entities

def decode_single(
    text: str,
    tokenizer,
    id2label: dict[int, str],
    predicted_ids_1d: np.ndarray,
    predicted_scores_1d: np.ndarray,
    offsets: Iterable[tuple[int, int]],
    special_tokens_mask: Iterable[int],
) -> list[dict[str, object]]:
    """1サンプルぶん（L次元）のBIO走査→スパン構築→隣接マージ。"""
    entities: list[dict[str, object]] = []
    state: dict[str, object] = {"label": None, "start": None, "end": None, "scores": []}

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
