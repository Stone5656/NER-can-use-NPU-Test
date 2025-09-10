"""
logger_utils.py
ログユーティリティ実装: 標準出力(色付き)とファイル出力をサポート。
Interface層(logger_interface.LoggerFactory)に従う。
"""

import logging
import sys
from ner_openvino.utils.logger_utils.logger_interface import LoggerFactory

# ANSI カラーコード定義
COLOR_RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[36m",   # Cyan
    logging.INFO: "\033[32m",    # Green
    logging.WARNING: "\033[33m", # Yellow
    logging.ERROR: "\033[31m",   # Red
    logging.CRITICAL: "\033[41m" # Red background
}

class ColorFormatter(logging.Formatter):
    """標準出力向けの色付きフォーマッタ（レベル名のみ色付け、ファイル名・行番号付き）"""
    def format(self, record: logging.LogRecord) -> str:
        # レベルに応じた色を選択
        color = COLORS.get(record.levelno, COLOR_RESET)

        # レベル名を色付きに差し替え
        original_levelname = record.levelname
        record.levelname = f"{color}{original_levelname}{COLOR_RESET}"

        # 通常のフォーマット
        log_fmt = (
            "%(asctime)s [%(levelname)s] %(name)s "
            "(%(filename)s:%(lineno)d): %(message)s"
        )
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        output = formatter.format(record)

        # レベル名を元に戻しておかないと他のハンドラに影響する
        record.levelname = original_levelname

        return output

def _get_logging(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    LoggerFactory 実装。
    - name: ロガー名
    - log_file: ファイル出力先パス。None の場合はファイル出力なし
    - level: ログレベル (デフォルト INFO)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # ハンドラ重複防止
    if logger.hasHandlers():
        logger.handlers.clear()

    # 標準出力（色付き + ファイル名/行番号）
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(ColorFormatter())
    logger.addHandler(stream_handler)

    # ファイル出力（色なし、標準フォーマット）
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

# この実装を Interface 用のエイリアスとして提供
LoggerFactoryImpl: LoggerFactory = _get_logging
