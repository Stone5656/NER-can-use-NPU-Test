import io
import os
import re
import sys
import logging
from contextlib import redirect_stdout
from ner_openvino.utils.logger_utils.logger_utils import LoggerFactoryImpl

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")  # ANSI エスケープのざっくり検出

def test_console_has_color_and_file_line(monkeypatch, tmp_path):
    # 標準出力をキャプチャ
    captured_stdout = io.StringIO()
    with redirect_stdout(captured_stdout):
        log_file_path = tmp_path / "test.log"
        logger = LoggerFactoryImpl("TEST_LOGGER", log_file=str(log_file_path), level=logging.DEBUG)

        # ログを出力（INFO と ERROR）
        logger.info("情報ログ: hello")
        logger.error("エラーログ: boom")

    console_output = captured_stdout.getvalue()

    # 1) コンソールに ANSI カラーが含まれる（INFO/ERROR どちらかでOK）
    assert ANSI_PATTERN.search(console_output), "標準出力に ANSI カラーコードが含まれていません"

    # 2) (filename:lineno) が含まれる
    #    例: "(test_logger_utils.py:XX):"
    assert re.search(r"\([^)]+:\d+\):", console_output), "標準出力に (filename:lineno) が含まれていません"

    # 3) ファイルが作成され、ANSI が含まれていない（プレーン）
    assert log_file_path.exists(), "ログファイルが作成されていません"
    file_text = log_file_path.read_text(encoding="utf-8")
    assert not ANSI_PATTERN.search(file_text), "ログファイルに ANSI カラーコードが含まれています（想定外）"

    # 4) ファイル側にも (filename:lineno) が残っている
    assert re.search(r"\([^)]+:\d+\):", file_text), "ログファイルに (filename:lineno) が含まれていません"

def test_interface_swap_with_mock(tmp_path):
    """
    Interface（LoggerFactory）に沿って Moc を差し替えられることを確認。
    """
    from ner_openvino.utils.logger_utils.logger_interface import LoggerFactory

    class MockLogger:
        def __init__(self):
            self.messages = []
        def info(self, message): self.messages.append(("INFO", message))
        def debug(self, message): self.messages.append(("DEBUG", message))
        def warning(self, message): self.messages.append(("WARNING", message))
        def error(self, message): self.messages.append(("ERROR", message))
        def critical(self, message): self.messages.append(("CRITICAL", message))

    def mock_factory(name: str, log_file: str | None = None, level: int = logging.INFO):
        return MockLogger()

    # 型的には LoggerFactory と互換
    mock_factory_typed: LoggerFactory = mock_factory  # Protocol適合チェック

    logger = mock_factory_typed("MOCK", log_file=None, level=logging.INFO)
    logger.info("mock info")
    logger.error("mock error")

    assert hasattr(logger, "messages")
    assert ("INFO", "mock info") in logger.messages
    assert ("ERROR", "mock error") in logger.messages
