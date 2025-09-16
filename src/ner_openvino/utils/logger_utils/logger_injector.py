# ner_openvino/utils/logger_utils/logger_injector.py
from __future__ import annotations
import os, logging, functools, inspect
from typing import Callable

# あなたの実装に合わせて適切な import に変えてください
from .logger_factory import LoggerFactoryImpl
from .level_mapper import map_level  # "INFO"→logging.INFO など


def _resolve_logger(name: str, log_file: str | None, level: int | str) -> logging.Logger:
    """既存ロガー優先で Logger を解決し、なければファクトリで生成して返す。

    優先順位:
        1) logging.getLogger(name) に既にハンドラが付いていれば、それをそのまま返す
        2) そうでなければ LoggerFactoryImpl(name, log_file, level) で新規生成

    パラメータ:
        name:
            ロガー名。パッケージ階層に合わせた階層名（例: "pkg.mod"）を推奨。
        log_file:
            ファイル出力を追加したい場合のパス。None の場合はファイル出力なし。
        level:
            ログレベル。数値（logging.DEBUG 等）/文字列（"DEBUG" 等）いずれも可。

    返り値:
        logging.Logger:
            既存の構成済みロガー、または新規生成したロガー。

    注意:
        - 既存ロガーのハンドラ存在有無で“構成済み”かどうかを判断します。
          他の箇所でハンドラを remove 済みの場合、毎回新規生成される点に留意。
        - Python の logging は基本的にスレッドセーフですが、root/親子ロガーと
          ハンドラの重複接続・propagate の設定によっては重複出力が起こり得ます。
    """
    existing = logging.getLogger(name)
    if existing.handlers:   # 既にどこかでハンドラ設定済み＝“既存のロガー”
        return existing
    return LoggerFactoryImpl(name, log_file=log_file, level=level)


def with_logger(
    name: str,
    log_file: str | None = None,
    env_var: str = "LOG_LEVEL",
) -> Callable:
    """ロガー依存を「見せずに」注入するデコレーター。

    目的:
        - 関数本体を変更せずに、実行時に logger を用意する。
        - 既存のグローバル/外部構成があるならそれを尊重し、無ければ生成する。
        - テストでは kwarg で任意の logger を渡して差し替えできるようにする。

    動作の流れ:
        A) 対象関数が引数 `logger` を受け取れる *かつ* 呼び出し側がそれを渡していない場合
           → kwargs["logger"] に注入して実行
        B) それ以外（= 関数シグネチャに logger 引数がない、または既に渡されている）
           → 関数が参照するモジュールのグローバル変数 `logger` にセットして実行
              （すでに `logger` が存在する場合は上書きしない）

    ロガー解決の優先順位:
        1) 既存ロガー（logging.getLogger(name) にハンドラあり）
        2) 新規生成（LoggerFactoryImpl(name, log_file, level)）
           - level は env_var（既定 "LOG_LEVEL"）から読み取った値を map_level で解決
           - env_var 未設定時は "INFO" として扱う

    引数:
        name:
            注入するロガー名。パッケージ構成に合わせると親子ロガーの伝播制御がしやすい。
        log_file:
            ファイル出力を追加したい場合のパス。None の場合はファイル出力なし。
        env_var:
            ログレベルを読み取る環境変数名（例: "LOG_LEVEL"）。未設定時は "INFO"。

    戻り値:
        Callable:
            デコレーター本体。

    例:
        >>> @with_logger(name="NER-OpenVINO-APP", log_file="logs/app.log")
        ... def f(x, *, logger):
        ...     logger.debug("debug message")
        ...     return x

        # 呼び出し側が logger を渡さなければ、環境変数 LOG_LEVEL に応じて注入される
        >>> f(1)

        # テストでは任意の fake/memory logger を差し込める
        >>> class Fake:  # 必要なメソッドだけ実装
        ...     def debug(self, *a, **k): pass
        >>> f(1, logger=Fake())

        >>> @with_logger("NER-OpenVINO-APP")
        ... def g(x):  # logger 引数を受けない関数
        ...     logger.info("use module-global logger")  # デコレーターが注入
        ...     return x

    設計ノート / 落とし穴:
        - “グローバル注入（B）”はプロセス内で共有されるため、並列テストや一部だけ
          別ロガーに切り替えたい場合には衝突し得ます。テスト容易性を重視する場合は
          可能なら `logger` を明示引数として受けられる形（A）を推奨。
        - 判定は `inspect.signature(func).parameters` を用いています。
    """
    def deco(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1) 関数が logger 引数を受け取れる場合は kwargs で注入
            params = inspect.signature(func).parameters
            if "logger" in params and "logger" not in kwargs:
                level = map_level(os.getenv(env_var, "INFO"))
                kwargs["logger"] = _resolve_logger(name, log_file, level)
                return func(*args, **kwargs)

            # 2) 受け取らない関数なら、関数が参照するモジュールグローバルに挿す
            global_scope = func.__globals__
            if global_scope.get("logger", None) is None:
                level = map_level(os.getenv(env_var, "INFO"))
                global_scope["logger"] = _resolve_logger(name, log_file, level)

            return func(*args, **kwargs)
        return wrapper
    return deco
