"""Download the SpaVista checkpoint and data into the repository layout."""

import argparse
import hashlib
import os
import shutil
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


REPO_ID = "ZNfiona1997/SpaVista"
CHECKPOINT_NAME = "SpaVistav1.pth"
CHECKPOINT_SHA256 = "8cb665b931341c21fe16ec7b7d455441869ba19deeedc6a901e4253ba71a83d9"
ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DATA_DIRS = (
    "data/SGE/her2st",
    "data/SGE/skin",
    "data/tumor_detection/TRISTdata",
    "data/protein/HEMIT",
    "data/HR/visiumHD_Breast_Cancer",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_checkpoint(repo_id: str) -> None:
    target = ROOT / "spavista" / CHECKPOINT_NAME
    if target.exists():
        if sha256(target) != CHECKPOINT_SHA256:
            raise RuntimeError(f"Existing checkpoint has a different SHA-256: {target}")
        print(f"Verified existing checkpoint: {target}")
        return

    files = HfApi().list_repo_files(repo_id, repo_type="model")
    candidates = [name for name in files if PurePosixPath(name).name == CHECKPOINT_NAME]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {CHECKPOINT_NAME} in {repo_id}; found {candidates}")
    source = Path(hf_hub_download(repo_id=repo_id, filename=candidates[0], repo_type="model"))
    if sha256(source) != CHECKPOINT_SHA256:
        raise RuntimeError("Downloaded checkpoint does not match the validated SHA-256.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pth.download")
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != CHECKPOINT_SHA256:
            raise RuntimeError("Checkpoint copy failed SHA-256 verification.")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved checkpoint: {target}")


def download_data(repo_id: str) -> None:
    files = HfApi().list_repo_files(repo_id, repo_type="model")
    if not any(name.startswith("data/") for name in files):
        raise RuntimeError(f"No data/ directory is published in {repo_id} yet.")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["data/**"],
        local_dir=str(ROOT),
    )
    missing = [name for name in REQUIRED_DATA_DIRS if not (ROOT / name).is_dir()]
    if missing:
        raise RuntimeError(f"Downloaded data are missing expected directories: {missing}")
    print(f"Downloaded data into: {ROOT / 'data'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID, help="Hugging Face model repository")
    parser.add_argument("--checkpoint", action="store_true", help="Download the SpaVista encoder checkpoint")
    parser.add_argument("--data", action="store_true", help="Download the data/ directory")
    args = parser.parse_args()
    if not args.checkpoint and not args.data:
        parser.error("Select --checkpoint, --data, or both.")
    if args.checkpoint:
        download_checkpoint(args.repo_id)
    if args.data:
        download_data(args.repo_id)


if __name__ == "__main__":
    main()
