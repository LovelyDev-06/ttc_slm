"""Small Hugging Face Hub helpers used for logs and resumable JSON checkpoints."""

import os
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError


def get_hub_api(config: dict) -> HfApi:
    return HfApi()


def ensure_repo(config: dict):
    repo_id = config["hub"]["repo_id"]
    create_repo(
        repo_id=repo_id,
        private=config["hub"].get("private", False),
        exist_ok=True,
        repo_type="model",
    )
    return repo_id


def push_file(local_path: str, config: dict, path_in_repo: str = None):
    api = get_hub_api(config)
    repo_id = ensure_repo(config)
    path_in_repo = path_in_repo or os.path.basename(local_path)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[hub_utils] pushed {local_path} -> {repo_id}/{path_in_repo}")


def download_file(config: dict, path_in_repo: str, local_path: str) -> bool:
    """
    Download a Hub file if it exists. Returns True when downloaded, False
    when there's nothing to restore yet — either because this specific file
    hasn't been pushed (EntryNotFoundError, e.g. mid-way through a fresh
    labeling run) or because the whole repo hasn't been created at all yet
    (RepositoryNotFoundError, e.g. the very first call of a brand-new repo,
    before any push_file() call has had a chance to create it). Treating
    only the first case as "no file yet" and letting the second crash was a
    real bug: a script that always calls download_file() before push_file()
    (checkpoint-then-create pattern) could never get past its very first
    run on a repo that doesn't exist yet.
    """
    try:
        cached = hf_hub_download(
            repo_id=config["hub"]["repo_id"],
            repo_type="model",
            filename=path_in_repo,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return False

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    with open(cached, "rb") as src, open(local_path, "wb") as dst:
        dst.write(src.read())
    print(f"[hub_utils] restored {path_in_repo} -> {local_path}")
    return True


def push_directory(local_dir: str, config: dict, path_in_repo: str = None):
    api = get_hub_api(config)
    repo_id = ensure_repo(config)
    api.upload_folder(
        folder_path=local_dir,
        path_in_repo=path_in_repo or os.path.basename(local_dir.rstrip("/")),
        repo_id=repo_id,
        repo_type="model",
    )