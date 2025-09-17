import pytest

from ner_openvino.utils.text_utils.split_longtext import split_text_into_chunks


@pytest.mark.parametrize(
    "text, n_chunks, expected_lengths",
    [
        ("abcdefgh", 4, [2, 2, 2, 2]),   # ぴったり割り切れる
        ("abcdefghi", 4, [3, 2, 2, 2]), # 端数あり → 空文字で埋める
        ("a", 4, [1, 0, 0, 0]),         # 1文字 → 残りは空文字
        ("", 4, [0, 0, 0, 0]),          # 空文字入力
    ]
)
def test_split_text_into_chunks(text, n_chunks, expected_lengths):
    chunks = split_text_into_chunks(text, n_chunks)

    # 必ず n_chunks 個返す
    assert len(chunks) == n_chunks

    # 各チャンクの長さが期待値通り
    assert [len(chunk) for chunk in chunks] == expected_lengths

    # 連結すると元の文字列に戻る
    assert "".join(chunks) == text
