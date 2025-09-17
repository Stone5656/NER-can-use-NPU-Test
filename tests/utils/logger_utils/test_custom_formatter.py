def test_color_formatter_no_side_effects_on_record(caplog):
    import logging
    from ner_openvino.utils.logger_utils.custom_formatter import ColorFormatter

    logger = logging.getLogger("NER-OpenVINO-APP")
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logger.addHandler(handler)

    with caplog.at_level(logging.INFO, logger="NER-OpenVINO-APP"):
        logger.info("hello")

    # LogRecord 自体の levelname が壊れていないか
    assert any(record.levelname == "INFO" for record in caplog.records)
