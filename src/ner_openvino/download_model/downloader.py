"""
models/downloader.py
Hugging Face Hub から選択ダウンロード（snapshot_download）を行う。
"""
from __future__ import annotations
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

from ner_openvino.download_model.config import DEFAULT_MODEL_REPO, load_allow_patterns, load_ignore_patterns
from ner_openvino.utils.logger_utils.logger_injector import with_logger


@with_logger("NER-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def download_model_snapshot(
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
    save_dir: Path | None = None,
    *,
    logger: logging.Logger,
) -> Path:
    # repo_id が None の場合は DEFAULT_MODEL_REPO を使う
    repo_id = repo_id or DEFAULT_MODEL_REPO

    allow_patterns = load_allow_patterns()
    ignore_patterns = load_ignore_patterns()

    logger.info(
        f"モデルの選択ダウンロードを開始: repo_id={repo_id}, revision={revision or 'default'} "
        f"(allow={len(allow_patterns)}件, ignore={len(ignore_patterns)}件)"
    )

    model_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=save_dir,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns if allow_patterns else None,
        ignore_patterns=ignore_patterns if ignore_patterns else None,
        local_files_only=False,
        token=token,
    )

    logger.info(f"モデルのスナップショット取得完了: {model_dir}")
    return Path(model_dir)
