from __future__ import annotations
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from dotenv import load_dotenv
import time
import asyncio

import openvino as ov
from fastapi import FastAPI, Depends, Request, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# === 追加: NPU ルート用ユーティリティ ===
from ner_openvino.npu_ner.decode_ner import decode_ner_outputs_batch
from ner_openvino.utils.text_utils.split_longtext import split_text_into_chunks
from ner_openvino.download_model.loader_intel import load_ner_model_intel
from src.ner_openvino.download_model.loader_intel_npu import load_npu_model_intel

load_dotenv()

# モデル保存先
SAVE_DIR = Path(
    os.getenv("NER_SAVE_DIR", "./models/tsmatz_intel")
).expanduser().resolve()

SAVE_DIR_NPU = Path(
    os.getenv("NER_SAVE_DIR_NPU", "./models/tsmatz_intel_npu")
).expanduser().resolve()

# 推論用の最大シーケンス長とバッチサイズ
max_seq_len = int(os.getenv("NER_MAX_SEQ_LEN", 256))
batch_size = int(os.getenv("NER_BATCH_SIZE", 4))

# アプリ起動時に使用するデバイス
init_device = str(
    os.getenv("NER_INIT_DEVICE", "AUTO")
)

class Entity(BaseModel):
    entity_group: str
    score: float
    word: str
    start: int
    end: int

class NERIn(BaseModel):
    text: str

class DeviceIn(BaseModel):
    device: str  # "CPU", "GPU", "NPU", "AUTO", "AUTO:NPU,CPU" など

# ------------ バックエンド抽象化 ------------
class BaseBackend:
    kind: Literal["pipeline", "npu"]
    device: str
    async def infer(self, text: str) -> list[dict]:
        raise NotImplementedError

class PipelineBackend(BaseBackend):
    def __init__(self, model_dir: Path, device: str = "AUTO"):
        # Transformers + OpenVINO（Optimum 連携 or ov_backend）での従来ルート
        ner = load_ner_model_intel(model_dir=model_dir)
        ner.model.to(device=device)  # AUTO/GPU/CPU など
        self._pipeline = pipeline(
            "token-classification",
            model=ner.model,
            tokenizer=ner.tokenizer,
            aggregation_strategy="simple",
        )
        self.kind = "pipeline"
        self.device = device

    async def infer(self, text: str) -> list[dict]:
        results = self._pipeline(text)
        for r in results:
            r["score"] = float(r["score"])
        return results

class NPUBackend(BaseBackend):
    def __init__(self, model_dir: Path, max_seq_len: int = 512, batch_size: int = 8):
        # ご提示の NPU ルート
        self._cfg = dict(max_seq_len=max_seq_len, batch_size=batch_size)
        self._ner = load_npu_model_intel(
            model_dir=model_dir,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
        )
        self.kind = "npu"
        self.device = "NPU"

    async def infer(self, text: str) -> list[dict]:
        # シンプルな分割→バッチ推論→結合。分割ごとに start/end をオフセット補正。
        chunks = split_text_into_chunks(text, n_chunks=self._cfg["batch_size"])
        # 空チャンクを落とす（decode 側が空配列返す前提でも安全のため）
        chunk_indices = [(i, c) for i, c in enumerate(chunks) if c]
        if not chunk_indices:  # 完全空文字
            return []

        texts = [c for _, c in chunk_indices]
        enc = self._ner.tokenizer(
            texts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self._cfg["max_seq_len"],
            return_offsets_mapping=False,
        )
        outputs = self._ner.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )

        entities_batch = decode_ner_outputs_batch(
            texts=texts,
            tokenizer=self._ner.tokenizer,
            id2label=self._ner.id2label,
            logits=outputs.logits,
            max_sequence_length=self._cfg["max_seq_len"],
        )

        # start/end を原文相対に補正して平坦化
        flat: list[dict] = []
        base = 0
        for (_, chunk), ents in zip(chunk_indices, entities_batch):
            for e in ents:
                # 1) ラベル名の取り出し（entity_group / label / entity に対応）
                raw = e.get("entity_group") or e.get("label") or e.get("entity")
                if not raw:
                    # 想定外のフォーマットはスキップ
                    continue
                # BIO プレフィックス除去（B-*, I-* → *）
                group = raw.split("-", 1)[1] if raw.startswith(("B-", "I-")) else raw

                # 2) 位置とスコア
                s_local = int(e.get("start", 0))
                t_local = int(e.get("end", 0))
                start = s_local + base
                end = t_local + base
                score = float(e.get("score", e.get("probability", 0.0)))

                # 3) 表示語（word）が無い場合は原文から復元
                word = e.get("word")
                if not word:
                    try:
                        # 'text' は infer(text: str) の引数
                        word = text[start:end]
                    except Exception:
                        word = ""

                flat.append({
                    "entity_group": group,
                    "score": score,
                    "word": word,
                    "start": start,
                    "end": end,
                })
            base += len(chunk)

        return flat


# ------------ FastAPI 本体 ------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 既定は AUTO で pipeline ルート
    app.state.backend = PipelineBackend(model_dir=SAVE_DIR, device=init_device)
    app.state.compile_lock = asyncio.Lock()
    yield

app = FastAPI(lifespan=lifespan)

def get_backend(request: Request) -> BaseBackend:
    return request.app.state.backend

BackendDep = Annotated[BaseBackend, Depends(get_backend)]

@app.get("/")
def health():
    devices = ov.Core().get_available_devices()
    return {
        "status": "ok",
        "devices": devices,
        "current_device": getattr(app.state.backend, "device", None),
        "backend": getattr(app.state.backend, "kind", None),
    }

@app.post("/ner_text", response_model=list[Entity])
async def ner_text(body: NERIn, backend: BackendDep):
    start_time = time.time()
    results = await backend.infer(body.text)
    end_time = time.time()
    print(f"{end_time - start_time:.5f}秒")
    return results

# -------- デバイス切替 API --------
@app.get("/devices")
def get_devices():
    core = ov.Core()
    return {
        "available": core.get_available_devices(),
        "current": getattr(app.state.backend, "device", None),
        "backend": getattr(app.state.backend, "kind", None),
    }

@app.post("/devices")
async def set_device(body: DeviceIn, request: Request):
    """
    例:
      {"device": "CPU"}
      {"device": "NPU"}
      {"device": "AUTO:NPU,CPU,GPU"}
    """
    device = body.device.strip()
    core = ov.Core()
    available = core.get_available_devices()

    async with request.app.state.compile_lock:
        try:
            # NPU 指定時の分岐
            if "NPU" in device.upper() and device.upper() == "NPU":
                if "NPU" not in available:
                    raise HTTPException(
                        status_code=400,
                        detail=f"NPU が利用できません（available={available}）。"
                    )
                # NPU 専用バックエンドへ切替
                request.app.state.backend = NPUBackend(
                    model_dir=SAVE_DIR_NPU,
                    max_seq_len=max_seq_len,
                    batch_size=batch_size,
                )
            else:
                # それ以外は pipeline 方式（AUTO/CPU/GPU/MULTI/AUTO:.. を含む）
                request.app.state.backend = PipelineBackend(
                    model_dir=SAVE_DIR,
                    device=device,
                )

            return {
                "message": "device set",
                "requested": device,
                "effective_device": request.app.state.backend.device,
                "backend": request.app.state.backend.kind,
                "available": available,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to switch to '{device}': {e}"
            ) from e
