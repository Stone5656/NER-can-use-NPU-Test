def test_get_logger_is_idempotent_for_handlers(tmp_path):
    from ner_openvino.utils.logger_utils.logger_factory import get_logger
    import logging

    logger_name = "NER-OpenVINO-APP"
    log_file_path = tmp_path / "app.log"

    first_logger = get_logger(logger_name, log_file=log_file_path, level="INFO")
    first_handler_count = len(first_logger.handlers)

    # 同じ名前・同じログファイルで再取得してもハンドラが増えないこと
    second_logger = get_logger(logger_name, log_file=log_file_path, level="DEBUG")
    second_handler_count = len(second_logger.handlers)

    assert first_logger is second_logger
    assert second_handler_count == first_handler_count
