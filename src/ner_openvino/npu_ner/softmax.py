"""softmax.py

モデルの出力（ロジット: まだ確率になっていない数値の集まり）を
「確率」として解釈するための関数をまとめています。

ここでは softmax 関数を使って、
各トークンに対して「一番確からしい分類番号」と
「そのときの確率」を計算します。

想定:
    - 入力は形状 (1, 長さ, 分類数) の3次元配列
    - この関数はバッチサイズ=1を前提にしているので、
      外側の次元は取り除いて (長さ, 分類数) として扱います
"""

from __future__ import annotations
import numpy as np
from ner_openvino.utils.logger_utils.logger_utils import LoggerFactoryImpl

logger = LoggerFactoryImpl("NER-OpenVINO-APP", log_file="logs/app.log")

# モデルの出力から確率と分類を出す
def compute_softmax(prediction_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """モデルが出した数値を softmax 関数で確率に変換し、
    各位置ごとに「一番確からしい分類番号」と「その確率」を返す。

    入力の形状は (1, L, C) を想定:
        - 1: バッチサイズ（ここでは1件分のみ）
        - L: トークン数（入力文を区切った数）
        - C: クラス数（分類できるラベルの数）

    Softmax の式:
        softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

    ポイント:
        - exp は指数関数（eのべき乗）
        - 数値が大きすぎると計算があふれるので、
          最大値を引いてから exp を計算する（オーバーフロー防止）

    Args:
        prediction_values (numpy.ndarray): 形状 (1, L, C) のロジット。

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - predicted_ids: 各位置で最尤の分類番号 (形状 (L,))
            - predicted_scores: そのときの確率 (形状 (L,))
    """
    # float32に変換
    values = prediction_values.astype(np.float32)
    logger.debug(f"float32に変換: shape={values.shape}, dtype={values.dtype}")

    # 各トークンの最大値を引いてオーバーフローを防ぐ
    max_per_token = values.max(axis=-1, keepdims=True)
    shifted = values - max_per_token
    logger.debug(f"最大値を減算: max.shape={max_per_token.shape}, "
                 f"shifted例={shifted[0,0,:5]}")

    # eのべき乗を計算して「正の値」に変換
    exp_values = np.exp(shifted)
    logger.debug(f"exp計算: shape={exp_values.shape}, exp例={exp_values[0,0,:5]}")

    # 各トークンごとに合計を出して「割り算の分母」にする
    sum_exp = exp_values.sum(axis=-1, keepdims=True)
    logger.debug(f"expの合計: shape={sum_exp.shape}, 例={sum_exp[0,0]}")

    # softmaxで確率を計算
    probabilities = exp_values / sum_exp  # (1, L, C)
    logger.debug(f"softmax確率: shape(before squeeze)={probabilities.shape}")

    # バッチ次元を取り除く
    probabilities = probabilities[0]  # (L, C)
    logger.debug(f"バッチ次元を削除: shape(after squeeze)={probabilities.shape}")

    # 各トークンの最尤ラベル番号
    predicted_ids = probabilities.argmax(axis=-1)   # (L,)
    logger.debug(f"最尤ラベルID: shape={predicted_ids.shape}, "
                 f"先頭5件={predicted_ids[:5]}")

    # 各トークンの最尤ラベルの確率
    predicted_scores = probabilities.max(axis=-1)   # (L,)
    logger.debug(f"ラベル確率: shape={predicted_scores.shape}, "
                 f"先頭5件={predicted_scores[:5]}")

    return predicted_ids, predicted_scores
