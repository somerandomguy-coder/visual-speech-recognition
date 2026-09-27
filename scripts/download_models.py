"""Download Auto-AVSR model weights and configuration artifacts."""

import os
from pathlib import Path
import urllib.request
import sys

FILES_TO_DOWNLOAD = [
    {
        "url": "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.json",
        "dest": "benchmarks/LRS3/language_models/lm_en_subword/model.json",
    },
    {
        "url": "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.pth",
        "dest": "benchmarks/LRS3/language_models/lm_en_subword/model.pth",
    },
    {
        "url": "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.json",
        "dest": "benchmarks/LRS3/models/LRS3_V_WER19.1/model.json",
    },
    {
        "url": "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth",
        "dest": "benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth",
    },
]

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"Already exists: {dest_path} ({dest_path.stat().st_size} bytes)")
        return
    print(f"Downloading {url} -> {dest_path} ...")
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            sys.stdout.write(f"\r  {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, str(dest_path), reporthook=report_progress)
    print("\n  Done.")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    for item in FILES_TO_DOWNLOAD:
        dest = repo_root / item["dest"]
        download_file(item["url"], dest)
    print("All Auto-AVSR model artifacts downloaded successfully!")

if __name__ == "__main__":
    main()
