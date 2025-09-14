from PIL import Image
from transformers import DonutProcessor
from optimum.intel.openvino import OVModelForVision2Seq

model_id = "oshizo/donut-base-japanese-visual-novel"  # 日本語で微調整済み（ドメイン特化）
processor = DonutProcessor.from_pretrained(model_id)

# 取得時にIRへ自動変換（初回は少し時間がかかる）
ov_model = OVModelForVision2Seq.from_pretrained(
    model_id,
    export=True,
    device="AUTO",  # "CPU"/"GPU"/"AUTO"など
)

# 任意：動作確認（DonutはJSONや文字列を生成するタスク定義が多い）
image = Image.open("sample_img/test_text.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
output_ids = ov_model.generate(pixel_values, max_new_tokens=256)
print(processor.batch_decode(output_ids, skip_special_tokens=True)[0])

# ローカル保存（IR一式 & プロセッサ）
save_dir = "./models/donut_ja_ov"
ov_model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print(f"Saved to: {save_dir}")
