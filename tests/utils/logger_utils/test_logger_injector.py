# tests/utils/logger_utils/test_logger_injector.py

def test_with_logger_injects_kwarg_and_respects_existing(caplog, monkeypatch):
    from ner_openvino.utils.logger_utils.logger_injector import with_logger
    import logging

    # 既存ロガー（ハンドラ付き）を用意
    logger_name = "NER-OpenVINO-APP"
    existing_logger = logging.getLogger(logger_name)
    if not existing_logger.handlers:
        existing_logger.addHandler(logging.StreamHandler())
    existing_logger.setLevel(logging.INFO)

    @with_logger(logger_name)
    def accept_logger_kwarg(*, logger):
        logger.info("injected via kwarg")

    with caplog.at_level(logging.INFO, logger=logger_name):
        accept_logger_kwarg()

    assert any(
        record.name == logger_name
        and record.levelno == logging.INFO
        and record.getMessage() == "injected via kwarg"
        for record in caplog.records
    )


def test_with_logger_global_fallback(caplog, monkeypatch):
    import logging
    import ner_openvino.utils.logger_utils.logger_injector as injector

    # caplog が拾えるよう、propagate=True・独自ハンドラなしのロガーを返す
    def make_caplog_compatible_logger(name, log_file=None, level=None):
        logger_for_caplog = logging.getLogger(name)
        logger_for_caplog.handlers.clear()
        logger_for_caplog.setLevel(logging.DEBUG)
        logger_for_caplog.propagate = True   # root へ流す（= caplog のハンドラが捕捉）
        return logger_for_caplog

    monkeypatch.setattr(injector, "_resolve_logger", make_caplog_compatible_logger)

    @injector.with_logger("NER-OpenVINO-APP")
    def without_logger_kwarg():
        # デコレータで module-global に注入される
        logger.debug("global injected for caplog")  # type: ignore[name-defined]

    with caplog.at_level(logging.DEBUG, logger="NER-OpenVINO-APP"):
        without_logger_kwarg()

    messages_from_target = [
        record.getMessage()
        for record in caplog.records
        if record.name == "NER-OpenVINO-APP" and record.levelno == logging.DEBUG
    ]
    assert "global injected for caplog" in messages_from_target
