import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from utils_extract import base_path_models


REPO_ID = "EleutherAI/pythia-6.9b"
REVISIONS = ["step0", "step256", "step512", "step2000", "step4000", "step16000", "step64000", "step143000"]


def materialize_snapshot(cache_dir: Path, snapshot_dir: Path) -> None:
    for src_path in snapshot_dir.iterdir():
        dest_path = cache_dir / src_path.name
        if os.path.lexists(dest_path):
            continue
        if src_path.is_dir():
            shutil.copytree(src_path, dest_path)
            continue
        try:
            os.link(src_path, dest_path)
        except OSError:
            shutil.copy2(src_path, dest_path)


def download_revision(repo_id: str, revision: str) -> None:
    cache_dir = Path(base_path_models) / repo_id / revision
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"→ Downloading {repo_id} @ {revision} into {cache_dir}")
    snapshot_dir = Path(
        snapshot_download(
            repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    )
    materialize_snapshot(cache_dir, snapshot_dir)


for revision in REVISIONS:
    download_revision(REPO_ID, revision)
