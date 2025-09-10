from pathlib import Path
from transformers import pipeline
from ner_openvino.download_model.loader_intel import load_ner_model_intel
import openvino as ov

save_dir = Path("./models/tsmatz_intel")

ner = load_ner_model_intel(model_dir=save_dir)

# NPUでコンパイル（未対応オペが心配なら AUTO:NPU を推奨）
ner.model.compile()
# 例: ner.model.compile(device="AUTO:NPU,CPU,GPU")

# デバイス確認
print(ov.Core().get_available_devices())  # ["CPU", "GPU", "NPU", ...] など

nlp = pipeline(
    "token-classification",
    model=ner.model,
    tokenizer=ner.tokenizer,
    aggregation_strategy="simple",
)

print(nlp(
    """Hugging Face配布モデルを使用する場合、商用可否や再配布条項は各モデルのLICENSEに依存。
    Notebookは技術的手順を示すだけなので、実運用前に必ず確認を。
    根拠：Notebook側は「外部公開モデルをOptimum経由で取得・変換する」
    流れを示すチュートリアルであり、利用条件はモデル元に準拠。"""))
