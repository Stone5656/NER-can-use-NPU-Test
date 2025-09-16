from pathlib import Path
from ner_openvino.npu_ner.decode_ner import decode_ner_outputs_batch
from ner_openvino.utils.text_utils.split_longtext import split_text_into_chunks
from src.ner_openvino.download_model.loader_intel_npu import load_npu_model_intel

model_dir = Path("./models/tsmatz_intel_npu")
max_sequence_length = 512
batch_size = 1
ner_models = load_npu_model_intel(
    model_dir=model_dir,
    max_seq_len=max_sequence_length,
    batch_size=batch_size
)

long_text = """拙者、親方と申すは、お立合いの中に、
御存知のお方も御座りましょうが、
お江戸を発って二十里上方、
相州小田原一色町をお過ぎなされて、
青物町を登りへおいでなさるれば、
欄干橋虎屋藤衛門、
只今は剃髪致して、円斎と名乗りまする。
元朝より大晦日まで、
お手に入れまする此の薬は、
昔、朕の国の唐人、
外郎という人、我が朝へ来り、
帝へ参内の折から、
この薬を深く籠め置き、用ゆる時は一粒ずつ、
冠のすき間より取り出す。
依ってその名を帝より、
「とうちんこう」と賜わる。
即ち文字には
「頂き、透く、香い」と書いて
「透頂香」と申す。
只今はこの薬、
殊の外世上に広まり、方々に偽看板を出し、
イヤ、小田原の、灰俵の、さん俵の、炭俵のと、
色々に申せども、
平仮名をもって「ういろう」と記せしは、
親方円斎ばかり。
もしやお立ち会いの中に、熱海か塔ノ沢へ湯治にお出でなさるるか、
又は伊勢御参宮の折からは、必ず門違いなされまするな。
お登りならば右の方、お下りなれば左側、
八方が八つ棟、表が三つ棟、玉堂造り、
破風には菊に桐のとうの御紋を御赦免あって、
系図正しき薬でござる。
＜第二節＞
いや、最前より家名の自慢ばかり申しても、
御存知ない方には、正身の胡椒の丸呑み、白河夜船、
さらば一粒食べかけて、
その気見合いをお目にかけましょう。
先ずこの薬をかように一粒舌の上にのせまして、
腹内へ納めますると……
いやぁどうも云えぬは、胃、心、肺、肝がすこやかになりて、
薫風咽より来り、口中微涼を生ずるが如し、
魚鳥、茸、麺類の食合わせ、其の他、万病速効ある事神の如し。
"""

texts = split_text_into_chunks(long_text, n_chunks=4)

enc = ner_models.tokenizer(
    texts,
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=max_sequence_length,  # ★ reshape時と一致
    return_offsets_mapping=False,    # ここでは不要（decode側で再トークナイズするため）
)

outputs = ner_models.model(
    input_ids=enc["input_ids"],
    attention_mask=enc["attention_mask"],
)

entities_batch = decode_ner_outputs_batch(
    texts=texts,
    tokenizer=ner_models.tokenizer,
    id2label=ner_models.id2label,
    logits=outputs.logits,
    max_sequence_length=max_sequence_length,
)

for i, ents in enumerate(entities_batch):
    print(f"=== Text {i} ===")
    for ent in ents:
        print(ent)
