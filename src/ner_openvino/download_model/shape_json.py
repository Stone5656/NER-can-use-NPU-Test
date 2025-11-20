from __future__ import annotations
import json
import os
from pathlib import Path


# ---------------------------------------------------------
# shape.json 読み書きユーティリティ
# ---------------------------------------------------------

def get_shape_json_path(model_dir: Path) -> Path:
    """
    環境変数 NER_SHAPE_JSON を使う。
    無ければ model_dir 無視で 4階層上の shape/shape.json を使う。
    """
    # デフォルトパス（4階層上 / shape / shape.json）
    default_path = "shape" 

    # getenv の第2引数に default_path をセット
    env_path = os.getenv("NER_SHAPE_JSON", default_path)

    path = Path(__file__).resolve().parents[3] / env_path / "shape.json"

    shape_path = Path(path)

    # 親ディレクトリを必ず作る
    shape_path.parent.mkdir(parents=True, exist_ok=True)

    return shape_path


def write_shape_json(path: Path, shape: dict) -> None:
    """shape.json を書き込み（初回は自動作成）"""
    with path.open("w", encoding="utf-8") as f:
        json.dump(shape, f, ensure_ascii=False, indent=2)


def read_shape_json(path: Path) -> dict | None:
    """shape.json を読み込み（なければ None）"""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_shape_same(path: Path, new_shape: dict) -> bool:
    """shape.json の shape と new_shape が一致するか判定"""
    old_shape = read_shape_json(path)
    if old_shape is None:
        return False
    return old_shape == new_shape
