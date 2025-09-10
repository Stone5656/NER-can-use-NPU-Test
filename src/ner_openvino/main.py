from ner_openvino.utils.logger_utils.logger_utils import LoggerFactoryImpl
from ner_openvino.download_model.loader import load_ner_model
from ner_openvino.download_model.downloader import download_model_snapshot

logger = LoggerFactoryImpl("NER-OpenVINO-APP", log_file="logs/app.log")

repo_id = "tsmatz/xlm-roberta-ner-japanese"
save_dir = "./models"

if __name__ == "__main__":
    # 1. 初回だけダウンロード
    model_dir = download_model_snapshot(
        repo_id=repo_id,
        save_dir=save_dir,
    )

    # 2. ローカルからロード
    ner = load_ner_model(model_dir=model_dir)

    logger.info(f"loaded from: {ner.model_dir}")
    logger.info(f"labels: {sorted(set(ner.id2label.values()))}")
