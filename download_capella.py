"""
Capella Space - Spotlight GEO Downloader (boto3 version)
=========================================================
Downloads Spotlight GEO GeoTIFF scenes directly from S3.
No AWS account needed. No AWS CLI needed. Pure Python.

Setup:
    pip install boto3 requests tqdm

Run:
    python download_capella.py

Files saved to: ./data/raw/capella_geo/
"""

import os
import json
import time
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# ─────────────────────────────────────────────────────────
#  CONFIG — edit these
# ─────────────────────────────────────────────────────────
SAVE_DIR     = "./data/raw/capella_geo"   # where .tif files are saved
MAX_SCENES   = 30                         # number of scenes to download
                                          # set to None to download ALL (~270)

BUCKET       = "capella-open-data"
PREFIX       = "data/"
REGION       = "us-west-2"

IMAGING_MODE = "SP"                       # SP=Spotlight
PRODUCT_TYPE = "GEO"                      # GEO=terrain-corrected GeoTIFF
POLARIZATION = "HH"                       # HH has the most scenes
EXTENSION    = ".tif"

LOG_FILE     = "./capella_downloaded.txt"
# ─────────────────────────────────────────────────────────


def get_s3_client():
    """Anonymous S3 client — no AWS account needed."""
    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(signature_version=UNSIGNED)
    )


def list_all_geo_files(s3, max_scenes=None):
    """List all SP GEO .tif files in the S3 bucket using paginator."""
    print(f"Scanning s3://{BUCKET}/{PREFIX}")
    print(f"Filtering for: {IMAGING_MODE}_{PRODUCT_TYPE}_{POLARIZATION}{EXTENSION}")
    print("(This takes 1-2 minutes — bucket has thousands of files)\n")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)

    matched = []
    total_scanned = 0

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            total_scanned += 1

            if total_scanned % 1000 == 0:
                print(f"  Scanned {total_scanned} files, found {len(matched)} matches...")

            filename = key.split("/")[-1]

            if (f"_{IMAGING_MODE}_" in filename and
                f"_{PRODUCT_TYPE}_" in filename and
                f"_{POLARIZATION}_" in filename and
                filename.endswith(EXTENSION)):

                size_mb = obj["Size"] / 1024 / 1024
                matched.append({
                    "key": key,
                    "filename": filename,
                    "size_mb": round(size_mb, 1)
                })
                print(f"  [FOUND #{len(matched)}] {filename} ({size_mb:.0f} MB)")

                if max_scenes and len(matched) >= max_scenes:
                    print(f"\nReached MAX_SCENES={max_scenes}, stopping.")
                    return matched

    print(f"\nTotal scanned: {total_scanned} files")
    return matched


def load_completed():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def mark_completed(filename):
    with open(LOG_FILE, "a") as f:
        f.write(filename + "\n")


def main():
    print("=" * 60)
    print("  Capella Space Spotlight GEO Downloader")
    print("=" * 60)
    print(f"  Imaging mode:  Spotlight (SP)")
    print(f"  Product type:  GEO (terrain-corrected GeoTIFF)")
    print(f"  Polarization:  {POLARIZATION}")
    print(f"  Max scenes:    {MAX_SCENES if MAX_SCENES else 'ALL'}")
    print(f"  Save dir:      {os.path.abspath(SAVE_DIR)}")
    print("=" * 60 + "\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Connecting to AWS S3 (anonymous, no account needed)...")
    s3 = get_s3_client()
    print("Connected.\n")

    files = list_all_geo_files(s3, max_scenes=MAX_SCENES)

    if not files:
        print("\nNo files found. Try changing POLARIZATION to 'VV' or check filters.")
        return

    total_gb = sum(f["size_mb"] for f in files) / 1024
    print(f"\n{'─'*60}")
    print(f"  Found:     {len(files)} files")
    print(f"  Total:     ~{total_gb:.1f} GB")
    print(f"  Est. time (50 Mbps): ~{total_gb*8/50/60:.0f} min")
    print(f"{'─'*60}\n")

    with open("./capella_manifest.json", "w") as f:
        json.dump(files, f, indent=2)
    print("Manifest saved: capella_manifest.json\n")

    completed  = load_completed()
    to_download = [f for f in files if f["filename"] not in completed]

    print(f"Already downloaded: {len(completed)}")
    print(f"Remaining:          {len(to_download)}\n")

    if not to_download:
        print("All files already downloaded!")
        return

    success = 0
    failed  = []

    for i, file in enumerate(to_download):
        filename   = file["filename"]
        key        = file["key"]
        size_mb    = file["size_mb"]
        local_path = os.path.join(SAVE_DIR, filename)

        print(f"[{i+1}/{len(to_download)}] {filename}  ({size_mb:.0f} MB)")

        if os.path.exists(local_path):
            print(f"  → already on disk, skipping\n")
            mark_completed(filename)
            success += 1
            continue

        start = time.time()
        try:
            s3.download_file(Bucket=BUCKET, Key=key, Filename=local_path)
            elapsed = time.time() - start
            speed   = size_mb / elapsed if elapsed > 0 else 0
            print(f"  → done in {elapsed:.0f}s  ({speed:.1f} MB/s)\n")
            mark_completed(filename)
            success += 1
        except Exception as e:
            if os.path.exists(local_path):
                os.remove(local_path)
            print(f"  → FAILED: {e}\n")
            failed.append(filename)

    print("=" * 60)
    print(f"  Succeeded: {success}")
    print(f"  Failed:    {len(failed)}")

    if failed:
        with open("./capella_failed.txt", "w") as f:
            f.write("\n".join(failed))
        print("  Failed list saved to: capella_failed.txt")
        print("  Re-run to retry automatically.")

    tif_files = list(Path(SAVE_DIR).glob("*.tif"))
    if tif_files:
        disk_gb = sum(f.stat().st_size for f in tif_files) / 1e9
        print(f"\n  Files on disk: {len(tif_files)}")
        print(f"  Disk used:     {disk_gb:.2f} GB")

    print(f"\n  Next step: python preprocess.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
