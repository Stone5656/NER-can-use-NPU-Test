# app.py
from __future__ import annotations
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Annotated
import time

import asyncio
import openvino as ov
from fastapi import FastAPI, Depends, Request, HTTPException
from pydantic import BaseModel
from transformers import pipeline

from ner_openvino.download_model.loader_intel import load_ner_model_intel

SAVE_DIR = Path("./models/tsmatz_intel")

class Entity(BaseModel):
    entity_group: str
    score: float
    word: str
    start: int
    end: int

class NERIn(BaseModel):
    text: str

class DeviceIn(BaseModel):
    device: str  # 例: "CPU", "GPU", "NPU", "AUTO", "AUTO:NPU,CPU"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    ner = load_ner_model_intel(model_dir=SAVE_DIR)

    # 初期はAUTO（環境に応じて最適に）
    ner.model.to(device="AUTO")

    # 共有状態
    app.state.ner_model = ner.model
    app.state.tokenizer = ner.tokenizer
    app.state.ner_pipeline = pipeline(
        "token-classification",
        model=app.state.ner_model,
        tokenizer=app.state.tokenizer,
        aggregation_strategy="simple",
    )
    app.state.compile_lock = asyncio.Lock()
    app.state.current_device = "AUTO"
    yield
    # --- shutdown ---

app = FastAPI(lifespan=lifespan)

def get_ner_pipeline(request: Request):
    return request.app.state.ner_pipeline

NERPipeline = Annotated[object, Depends(get_ner_pipeline)]

@app.get("/")
def health():
    devices = ov.Core().get_available_devices()
    return {"status": "ok", "devices": devices, "current_device": getattr(app.state, "current_device", None)}

@app.post("/ner_text", response_model=list[Entity])
def ner_text(body: NERIn, nlp=Depends(get_ner_pipeline)):
    start_time = time.time()
    results = nlp(body.text)
    for r in results:
        # numpy.float32 → Python float
        r["score"] = float(r["score"])
    end_time = time.time()
    print(f"{end_time - start_time:.5f}秒")
    return results

# -------- ここから set_device 用 API --------

@app.get("/devices")
def get_devices():
    core = ov.Core()
    return {
        "available": core.get_available_devices(),  # 例: ["CPU", "GPU", "NPU"]
        "current": getattr(app.state, "current_device", None),
    }

@app.post("/devices")
async def set_device(body: DeviceIn, request: Request):
    """
    デバイスを切替える。例:
      {"device": "CPU"}
      {"device": "NPU"}
      {"device": "AUTO:NPU,CPU,GPU"}
    """
    device = body.device.strip()
    async with request.app.state.compile_lock:
        try:
            # 1) モデルを再コンパイル
            request.app.state.ner_model.to(device=device)

            # 2) pipeline を作り直す（安全のため毎回再生成）
            request.app.state.ner_pipeline = pipeline(
                "token-classification",
                model=request.app.state.ner_model,
                tokenizer=request.app.state.tokenizer,
                aggregation_strategy="simple",
            )
            request.app.state.current_device = device

            # 3) 実際にどのデバイスが使われているかは AUTO の場合見えづらいので available も返す
            available = ov.Core().get_available_devices()
            return {"message": "device set", "requested": device, "available": available}
        except Exception as e:
            # 無効な指定や未対応演算子などはここで 400
            raise HTTPException(status_code=400, detail=f"Failed to compile on device '{device}': {e}") from e
