from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, ModelInfo, hf_hub_url, snapshot_download
from ner_openvino.download_model.config import DEFAULT_MODEL_REPO, load_allow_patterns, load_ignore_patterns
from ner_openvino.download_model.downloader import get_model_urls
from ner_openvino.hf_paths import filter_repo_objects


def get_model_urls(
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    # repo_id が None の場合は DEFAULT_MODEL_REPO を使う
    repo_id = repo_id or DEFAULT_MODEL_REPO

    allow_patterns = load_allow_patterns()
    ignore_patterns = load_ignore_patterns()

    api = HfApi(
        token=token,
    )
    repo_info: ModelInfo = api.repo_info(
        repo_id=repo_id, repo_type="model", revision=revision)
    repo_files: Iterable[str] = [
        f.rfilename for f in repo_info.siblings] if repo_info.siblings is not None else []
    filtered_repo_files: Iterable[str] = filter_repo_objects(
        items=repo_files,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    urls = []
    for f in filtered_repo_files:
        urls.append(
            hf_hub_url(
                repo_id=repo_id,
                filename=f,
                revision=revision,
                repo_type="model",
            )
        )
    print(urls)


if __name__ == "__main__":
    get_model_urls()
