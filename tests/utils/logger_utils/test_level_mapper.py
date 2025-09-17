def test_map_level_basic():
    from ner_openvino.utils.logger_utils.level_mapper import map_level
    import logging
    assert map_level("debug") == logging.DEBUG
    assert map_level("INFO") == logging.INFO
    assert map_level(10) == logging.DEBUG
    assert map_level("unknown") == logging.INFO  # 既定フォールバック
