from pathlib import Path

from ner_openvino.utils.text_utils.read_line import clean_lines

def test_clean_lines_excludes_comments_and_empty_lines(tmp_path: Path):
    # 一時ファイルを作成
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        """
        # コメント行
        有効な行1
          
        有効な行2
        # another comment
        """,
        encoding="utf-8"
    )

    result = clean_lines(str(file_path))
    assert result == ["有効な行1", "有効な行2"]


def test_clean_lines_returns_empty_list_if_file_not_exists(tmp_path: Path):
    fake_path = tmp_path / "not_exist.txt"
    assert clean_lines(str(fake_path)) == []
