"""
uv run ocr_pipeline_donut.py --model models/donut_ja_ov --input ./sample_img/test_sentense.png --no-line-split
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from transformers import DonutProcessor
from optimum.intel.openvino import OVModelForVision2Seq


# -----------------------
# 画像前処理ユーティリティ
# -----------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def deskew(gray: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    # 二値化 → 膨張でテキスト塊を強調 → 最小外接矩形の角度で補正
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mor = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    coords = np.column_stack(np.where(mor > 0))
    if coords.size == 0:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # OpenCVは[-90,0)で返る。水平近傍に正規化
    if angle < -45:
        angle = 90 + angle

    if abs(angle) > max_angle:
        # 角度が異常に大きい場合は補正無しとする
        return gray

    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def denoise(gray: np.ndarray) -> np.ndarray:
    # 軽いノイズ低減（文字のエッジを残しやすい）
    return cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)

def adaptive_binarize(gray: np.ndarray, enable: bool) -> np.ndarray:
    if not enable:
        return gray
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

def pad_to_square(img: np.ndarray, pad_value: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    # 白背景
    canvas = np.full((size, size), pad_value, dtype=img.dtype) if img.ndim == 2 else \
             np.full((size, size, 3), pad_value, dtype=img.dtype)
    y = (size - h) // 2
    x = (size - w) // 2
    canvas[y:y+h, x:x+w] = img
    return canvas

def resize_for_donut(img_rgb: np.ndarray, target: int) -> np.ndarray:
    # target は Donut の入力推奨サイズ（例：480 もしくは 640）
    return cv2.resize(img_rgb, (target, target), interpolation=cv2.INTER_CUBIC)

# -----------------------
# 行セグメンテーション（横書き想定）
# -----------------------
def segment_lines(gray: np.ndarray, min_height: int = 12) -> List[Tuple[int, int, int, int]]:
    """
    return: list of (x, y, w, h) for each line region (top→bottom順)
    """
    bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # 横方向に連結しやすいカーネルでモルフォロジ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gray.shape[1] // 50 + 1, 3))
    connected = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h >= min_height:
            boxes.append((x, y, w, h))
    # 上から順に
    boxes.sort(key=lambda b: b[1])
    return boxes

# -----------------------
# Donut 推論
# -----------------------
class DonutOVOCR:
    def __init__(self, model_dir_or_id: str, device: str = "AUTO"):
        # 取得（必要なら export=True でIR化しつつ読み込み可）
        self.processor = DonutProcessor.from_pretrained(model_dir_or_id)
        self.model = OVModelForVision2Seq.from_pretrained(model_dir_or_id, device=device)
        # 画像サイズの推奨値（DonutProcessor の設定を尊重）
        size = getattr(self.processor.image_processor, "size", None)
        # size は dict のことがある（{"height":480,"width":480}等）
        if isinstance(size, dict):
            self.target_size = int(size.get("height", 640))
        elif isinstance(size, (tuple, list)):
            self.target_size = int(size[0])
        else:
            self.target_size = 640  # フォールバック

    def infer_image(self, img_bgr: np.ndarray, per_line: bool = True, binarize: bool = False,
                    max_new_tokens: int = 256, decode_skip_special_tokens: bool = True) -> str:
        # ---- 前処理（共通） ----
        gray = to_gray(img_bgr)
        gray = deskew(gray)
        gray = enhance_contrast(gray)
        gray = denoise(gray)
        gray_for_seg = gray.copy()
        gray = adaptive_binarize(gray, enable=binarize)

        if not per_line:
            # 単発（1枚まるごと）
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            rgb = pad_to_square(rgb)
            rgb = resize_for_donut(rgb, self.target_size)
            pil = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            pixel_values = self.processor(images=pil, return_tensors="pt").pixel_values
            out_ids = self.model.generate(pixel_values, max_new_tokens=max_new_tokens)
            text = self.processor.batch_decode(out_ids, skip_special_tokens=decode_skip_special_tokens)[0]
            return text.strip()

        # ---- 行セグメントごとに OCR ----
        lines = segment_lines(gray_for_seg)
        results = []
        for (x, y, w, h) in lines:
            roi = gray[y:y+h, x:x+w]
            rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            rgb = pad_to_square(rgb)
            rgb = resize_for_donut(rgb, self.target_size)
            pil = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            pixel_values = self.processor(images=pil, return_tensors="pt").pixel_values
            out_ids = self.model.generate(pixel_values, max_new_tokens=max_new_tokens)
            decoded = self.processor.batch_decode(out_ids, skip_special_tokens=decode_skip_special_tokens)[0]
            results.append(decoded.strip())

        return "\n".join([r for r in results if r])

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="OCR pipeline with Donut + OpenVINO")
    ap.add_argument("--model", type=str, required=True,
                    help="Hugging Face model id or local dir (IR/processorが保存されている場所)")
    ap.add_argument("--input", type=str, required=True, help="画像ファイル or ディレクトリ")
    ap.add_argument("--device", type=str, default="AUTO", help="CPU/GPU/AUTO など")
    ap.add_argument("--no-line-split", action="store_true", help="行分割を無効化（1枚まとめて推論）")
    ap.add_argument("--binarize", action="store_true", help="自動二値化を有効化")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    engine = DonutOVOCR(args.model, device=args.device)

    in_path = Path(args.input)
    paths = [in_path] if in_path.is_file() else sorted(
        [p for p in in_path.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
    )

    if not paths:
        print("No images found.")
        return

    t0 = time.time()
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] cannot read: {p}")
            continue
        text = engine.infer_image(
            img_bgr=img,
            per_line=not args.no_line_split,
            binarize=args.binarize,
            max_new_tokens=args.max_new_tokens,
        )
        print("=" * 80)
        print(p.name)
        print("-" * 80)
        print(text)

    print("=" * 80)
    print(f"Done. {len(paths)} file(s) processed in {time.time() - t0:.2f}s.")

if __name__ == "__main__":
    main()
