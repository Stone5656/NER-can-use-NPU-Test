from __future__ import annotations

def split_text_into_chunks(text: str, n_chunks: int = 4) -> list[str]:
    """文字数に基づいてテキストを n_chunks に分割する。
    足りない要素は空文字 "" で補う。

    Args:
        text (str): 入力テキスト
        n_chunks (int): 分割数（デフォルト4）

    Returns:
        list[str]: 分割後のテキストリスト（長さ n_chunks）
    """
    if n_chunks <= 0:
        raise ValueError("n_chunks must be >= 1")

    length = len(text)
    if length == 0:
        # 空文字なら n_chunks 個すべて "" を返す
        return [""] * n_chunks

    q, r = divmod(length, n_chunks)
    sizes = [q + 1 if i < r else q for i in range(n_chunks)]

    chunks = []
    idx = 0
    for size in sizes:
        if size > 0:
            chunks.append(text[idx:idx+size])
            idx += size
        else:
            chunks.append("")

    return chunks
