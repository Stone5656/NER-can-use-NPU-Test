"""
utils/text_utils/read_line.py
テキストファイルから行を読み取り、空行・コメント(#...)を除去して返す。
"""
from __future__ import annotations
from pathlib import Path

def clean_lines(path_str: str) -> list[str]:
    """
    テキストファイルを読み込み、各行から改行を削除して返す。
    #が文頭にある場合はその行を読み込まない

    Args:
        path (str): ファイルパス

    Returns:
        list[str]: 整形済みの行リスト
    """
    text_path = Path(path_str)
    if not text_path.exists():
        return []

    result: list[str] = []

    for raw in text_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        result.append(line)
    return result
