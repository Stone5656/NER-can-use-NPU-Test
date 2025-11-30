# ner_openvino/download_model/device_enum.py
from __future__ import annotations
from enum import Enum


class Device(str, Enum):
    AUTO = "AUTO"
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"

    @classmethod
    def from_str(cls, value: str) -> "Device":
        """
        runtime で string が来たときも、ここを通せば Enum に正規化できる。
        想定外の値なら ValueError。
        """
        return cls(value.upper())
