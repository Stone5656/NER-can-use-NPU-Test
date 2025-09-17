# custom_formatter.py
import logging, sys
from .constants import COLOR_RESET, COLORS

class ColorFormatter(logging.Formatter):
    """色付き: レベル名だけ色付け。ファイル名/行番号付き"""
    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelno, COLOR_RESET)
        original = record.levelname
        record.levelname = f"{color}{original}{COLOR_RESET}"

        fmt = "%(name)s [%(levelname)s] (%(filename)s:%(lineno)d): %(message)s"
        formatter = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")
        out = formatter.format(record)

        record.levelname = original
        return out

def build_stream_handler(level: int) -> logging.Handler:
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(level)
    h.setFormatter(ColorFormatter())
    return h

def build_file_handler(log_file: str, level: int) -> logging.Handler:
    h = logging.FileHandler(log_file, encoding="utf-8")
    h.setLevel(level)
    fmt = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    h.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    return h
