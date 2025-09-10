"""
logger_interface.py
ロガー取得のインターフェース定義。
テスト時にはここを Moc 実装に差し替え可能。
"""

import logging
from typing import Protocol

class LoggerFactory(Protocol):
    def __call__(self, name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
        """ロガーを生成して返す"""
