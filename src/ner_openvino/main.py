from ner_openvino.utils.logger_utils.logger_utils import LoggerFactoryImpl
from ner_openvino.download_model.loader import load_ner_model

logger = LoggerFactoryImpl("NER-OpenVINO-APP", log_file="logs/app.log")
save_dir = "./models"

if __name__ == "__main__":
    ner = load_ner_model(
        save_dir=save_dir
    )
    logger.info(f"loaded from: {ner.model_dir}")
    logger.info(f"labels: {sorted(set(ner.id2label.values()))}")
