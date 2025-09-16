# tests/utils/logger_utils/conftest.py
import logging
import pytest

@pytest.fixture(autouse=True)
def _route_app_logger_for_caplog(monkeypatch):
    name = "NER-OpenVINO-APP"
    logger = logging.getLogger(name)

    # 退避
    old_level = logger.level
    old_handlers = logger.handlers[:]
    old_prop = logger.propagate

    # caplog が root に付けたハンドラで拾えるようにする
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    yield

    # 復元
    logger.handlers[:] = old_handlers
    logger.setLevel(old_level)
    logger.propagate = old_prop
