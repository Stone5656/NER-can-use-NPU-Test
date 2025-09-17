"""softmax.py

モデルの出力（ロジット: まだ確率になっていない数値の集まり）を
「確率」として解釈するための関数をまとめています。

ここでは softmax 関数を使って、
各トークンに対して「一番確からしい分類番号」と
「そのときの確率」を計算します。

想定:
    - 入力は形状 (バッチサイズ, 長さ, 分類数) の3次元配列
"""

from __future__ import annotations
import logging
import numpy as np
from ner_openvino.utils.logger_utils.logger_injector import with_logger


# モデルの出力から確率と分類を出す
@with_logger("NER-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_var="LOG_LEVEL")
def compute_softmax_batch(
    prediction_values: np.ndarray,
    *,
    logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray]:
    """モデルが出した数値を softmax 関数で確率に変換し、
    各位置ごとに「一番確からしい分類番号」と「その確率」を返す。

    入力の形状は (B, L, C) を想定:
        - B: バッチサイズ
        - L: トークン数（入力文を区切った数）
        - C: クラス数（分類できるラベルの数）

    Softmax の式:
        softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

    ポイント:
        - exp は指数関数（eのべき乗）
        - 数値が大きすぎると計算があふれるので、
          最大値を引いてから exp を計算する（オーバーフロー防止）

    Args:
        prediction_values (numpy.ndarray): 形状 (B, L, C) のロジット。

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - predicted_ids: 各位置で最尤の分類番号 (形状 (B, L,))
            - predicted_scores: そのときの確率 (形状 (B, L,))
    """
    if prediction_values.ndim != 3:
        raise ValueError(f"expected (B, L, C), got {prediction_values.shape}")

    # float32に変換
    values = prediction_values.astype(np.float32)
    logger.debug(f"float32に変換: shape={values.shape}, dtype={values.dtype}")

    max_per_token = values.max(axis=-1, keepdims=True)
    shifted = values - max_per_token

    B, L, C = values.shape
    if B and L and C:
        logger.debug(
            f"最大値を減算: max.shape={max_per_token.shape}, "
            f"shifted例={shifted[0, 0, :min(5, C)]}"
        )

    # softmaxで確率を計算
    exp_values = np.exp(shifted)
    sum_exp = exp_values.sum(axis=-1, keepdims=True)
    probabilities = exp_values / sum_exp  # (B, L, C)

    predicted_ids = probabilities.argmax(axis=-1)   # (B, L)
    predicted_scores = probabilities.max(axis=-1)   # (B, L)

    return predicted_ids, predicted_scores
