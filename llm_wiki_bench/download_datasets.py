#!/usr/bin/env python3
"""Download public multi-hop QA evaluation datasets.

Supported datasets:
- HotpotQA (distractor setting, dev set)
- MuSiQue-Ans (dev set)
- 2WikiMultiHopQA (dev set)

Usage:
    python download_datasets.py                     # download all
    python download_datasets.py --dataset hotpotqa
    python download_datasets.py --dataset musique
    python download_datasets.py --dataset 2wikimhqa
"""

import argparse
import json
import os
import sys
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets"


def download_hotpotqa(output_dir: Path):
    """Download the HotpotQA distractor dev set.

    Source: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    """
    import urllib.request

    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    out_path = output_dir / "hotpotqa_dev.json"

    if out_path.exists():
        print(f"  ✅ HotpotQA dev already exists: {out_path}")
        return out_path

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇️  Downloading HotpotQA dev set...")
    print(f"     URL: {url}")

    try:
        urllib.request.urlretrieve(url, str(out_path))
        # Validate.
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  ✅ HotpotQA dev: {len(data)} samples -> {out_path}")
        return out_path
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        if out_path.exists():
            out_path.unlink()
        return None


def download_musique(output_dir: Path):
    """Download the MuSiQue-Ans dev set.

    Source: https://github.com/StonyBrookNLP/musique
    """
    import urllib.request
    import zipfile

    # MuSiQue GitHub release URL.
    url = "https://github.com/StonyBrookNLP/musique/raw/main/data/musique_ans_v1.0_dev.jsonl"
    out_path = output_dir / "musique_dev.jsonl"

    if out_path.exists():
        print(f"  ✅ MuSiQue dev already exists: {out_path}")
        return out_path

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇️  Downloading MuSiQue dev set...")
    print(f"     URL: {url}")

    try:
        urllib.request.urlretrieve(url, str(out_path))
        # Validate.
        count = 0
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        print(f"  ✅ MuSiQue dev: {count} samples -> {out_path}")
        return out_path
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        print(f"  [warn] MuSiQue may need to be downloaded manually:")
        print(f"     1. Visit https://github.com/StonyBrookNLP/musique")
        print(f"     2. Download data/musique_ans_v1.0_dev.jsonl")
        print(f"     3. Place it at {output_dir}/musique_dev.jsonl")
        if out_path.exists():
            out_path.unlink()
        return None


def download_2wikimhqa(output_dir: Path):
    """Download the 2WikiMultiHopQA dev set.

    Source: https://github.com/Alab-NII/2wikimultihop
    """
    import urllib.request

    url = "https://github.com/Alab-NII/2wikimultihop/raw/main/data/dev.json"
    out_path = output_dir / "2wikimhqa_dev.json"

    if out_path.exists():
        print(f"  ✅ 2WikiMultiHopQA dev already exists: {out_path}")
        return out_path

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇️  Downloading 2WikiMultiHopQA dev set...")
    print(f"     URL: {url}")

    try:
        urllib.request.urlretrieve(url, str(out_path))
        # Validate.
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  ✅ 2WikiMultiHopQA dev: {len(data)} samples -> {out_path}")
        return out_path
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        print(f"  [warn] 2WikiMultiHopQA may need to be downloaded manually:")
        print(f"     1. Visit https://github.com/Alab-NII/2wikimultihop")
        print(f"     2. Download data/dev.json")
        print(f"     3. Place it at {output_dir}/2wikimhqa_dev.json")
        if out_path.exists():
            out_path.unlink()
        return None


DATASET_DOWNLOADERS = {
    "hotpotqa": download_hotpotqa,
    "musique": download_musique,
    "2wikimhqa": download_2wikimhqa,
}


def main():
    parser = argparse.ArgumentParser(description="Download multi-hop QA evaluation datasets")
    parser.add_argument("--dataset", "-d", choices=list(DATASET_DOWNLOADERS.keys()),
                        help="Dataset to download (omit to download all)")
    parser.add_argument("--output-dir", "-o", type=str, default=str(DATASETS_DIR),
                        help=f"Output directory (default: {DATASETS_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Multi-hop QA dataset download")
    print("=" * 60)

    if args.dataset:
        datasets_to_download = {args.dataset: DATASET_DOWNLOADERS[args.dataset]}
    else:
        datasets_to_download = DATASET_DOWNLOADERS

    results = {}
    for name, downloader in datasets_to_download.items():
        print(f"\n  Fetching {name}:")
        ds_dir = output_dir / name
        result = downloader(ds_dir)
        results[name] = "ok" if result else "fail"

    print(f"\n{'=' * 60}")
    print("  Download results")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:15s}  {status}")


if __name__ == "__main__":
    main()
