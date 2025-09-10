"""
import_test.py
このファイルでは現在の環境に optimum, openvino, nncf, transformers などの
今後使用するライブラリ群がインストールされているかを確認し、
それぞれのバージョン情報を出力します。

- distribution（パッケージ）としてのバージョン確認
- module（import可能かどうか）の確認
- optimum が namespace package かどうかの確認
"""

from importlib.metadata import version as get_distribution_version, PackageNotFoundError
import importlib
import pkgutil
import sys
from ner_openvino.utils.logger_utils.logger_utils import LoggerFactoryImpl

logger = LoggerFactoryImpl("NER-OpenVINO-APP", log_file="logs/app.log")

def get_distribution_version_safe(distribution_name: str) -> str:
    """指定した distribution のバージョンを返す。存在しなければ 'not found' を返す。"""
    try:
        return get_distribution_version(distribution_name)
    except PackageNotFoundError:
        return "not found"

def get_module_version_safe(module_name: str) -> str:
    """指定した module を import し、__version__ 属性を返す。なければ説明文を返す。"""
    try:
        imported_module = importlib.import_module(module_name)
        module_version = getattr(imported_module, "__version__", "(namespace または __version__ 未定義)")
        return module_version
    except Exception as exception:
        return f"import error: {exception}"

# 確認対象の distribution 名一覧
distribution_names = [
    "optimum",
    "optimum-intel",
    "openvino",
    "nncf",
    "transformers",
    "huggingface_hub",
]

# 確認対象の module 名一覧
module_names = [
    "optimum",
    "optimum.intel", 
    "optimum.intel.openvino",
    "openvino",
    "nncf",
    "transformers",
    "huggingface_hub",
]

logger.info("== Distribution のバージョン確認 ==")
for distribution_name in distribution_names:
    distribution_version = get_distribution_version_safe(distribution_name)
    logger.info(f"{distribution_name:16s} -> {distribution_version}")

logger.info("== Module の import チェック ==")
for module_name in module_names:
    module_version = get_module_version_safe(module_name)
    logger.info(f"{module_name:20s} -> {module_version}")

# optimum が namespace package かどうかを確認
try:
    import optimum
    logger.debug("optimum.__file__:", getattr(optimum, "__file__", None))
    logger.debug("sys.path[0]       :", sys.path[0])
    if getattr(optimum, "__file__", None) is None:
        logger.debug("optimum は namespace package として認識されています")
except Exception as exception:
    logger.debug("optimum の import に失敗:", exception)

# サブモジュール探索（デバッグ用）
try:
    import optimum
    logger.debug("optimum 配下のサブモジュール一覧:")
    for module_info in pkgutil.walk_packages(optimum.__path__, prefix="optimum."):
        logger.info(f"  - {module_info.name}")
except Exception:
    pass
