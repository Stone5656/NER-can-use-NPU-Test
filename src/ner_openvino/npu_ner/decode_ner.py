from __future__ import annotations
import numpy as np

from ner_openvino.npu_ner.decryption import decode_single
from ner_openvino.npu_ner.softmax import compute_softmax_batch


def decode_ner_outputs_batch(
    texts: list[str] | str,
    tokenizer,
    id2label: dict[int, str],
    logits: np.ndarray,
    max_sequence_length: int,
) -> list[list[dict[str, object]]]:
    """モデルのロジット (B, L, C) を、サンプルごとのエンティティ配列に復号する（バッチ対応）。

    入力:
        texts:
            - list[str] のとき: バッチ内テキスト列（長さ B）
            - str のとき: 1 件のテキストとして扱う（B=1）
        tokenizer:
            - Hugging Face の Fast 系推奨。`offset_mapping` と `special_tokens_mask` を返せること。
        id2label:
            - ラベルID→ラベル名の辞書
        logits:
            - 形状 (B, L, C) のロジット（float系推奨）
        max_sequence_length:
            - トークナイズ長（NPU 等で固定長の場合はその長さ）

    出力:
        list[list[dict]]: バッチ各要素（長さ B）それぞれのエンティティ配列
    """
    # --- 1) 入力正規化・検証 ---
    # 1テキスト(str)が来たらB=1として扱う
    if isinstance(texts, str):
        texts_list: list[str] = [texts]
    else:
        texts_list = list(texts)

    if logits.ndim != 3:
        raise ValueError(f"Expected logits shape to be (B, L, C), got {logits.shape}")

    B, L, C = logits.shape
    if len(texts_list) != B:
        raise ValueError(f"len(texts) ({len(texts_list)}) must equal batch size B ({B}).")

    # --- 2) 一括トークナイズ（固定長でパディング/切り詰め） ---
    tokenization = tokenizer(
        texts_list,
        return_tensors=None,             # Pythonのlistで受ける
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets_batch = tokenization["offset_mapping"]      # List[List[Tuple[int,int]]]（各長さ ~ L）
    special_batch = tokenization["special_tokens_mask"] # List[List[int]]（各長さ ~ L）

    # --- 3) softmax + argmax をバッチで計算 ---
    # 期待する戻り値の形状: (B, L), (B, L)
    predicted_ids_2d, predicted_scores_2d = compute_softmax_batch(logits)

    # --- 4) サンプルごとに _decode_single へ橋渡し ---
    results: list[list[dict[str, object]]] = []
    for b in range(B):
        # tokenizer側とlogits側で L がずれていたら早期に検知
        if len(offsets_batch[b]) != L or len(special_batch[b]) != L:
            raise ValueError(
                f"Tokenized length (offsets={len(offsets_batch[b])}, specials={len(special_batch[b])}) "
                f"must match logits length L ({L}) for sample index {b}."
            )

        entities = decode_single(
            text=texts_list[b],
            tokenizer=tokenizer,
            id2label=id2label,
            predicted_ids_1d=predicted_ids_2d[b],       # shape: (L,)
            predicted_scores_1d=predicted_scores_2d[b], # shape: (L,)
            offsets=offsets_batch[b],                   # 長さ L
            special_tokens_mask=special_batch[b],       # 長さ L
        )
        results.append(entities)

    return results
