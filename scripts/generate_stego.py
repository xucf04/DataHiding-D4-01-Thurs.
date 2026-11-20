"""Generate reproducible S-UNIWARD 0.4 bpp stego PGM files on Windows.

Run with --limit 5 first, then --all for both complete datasets.
The official executable simulates optimal embedding when STC height is zero.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "vendor" / "S-UNIWARD_windows_vc_2010_v10" / "executable" / "S-UNIWARD.exe"
SOURCE_ROOT = ROOT / "data" / "cover_256"
TARGET_ROOT = ROOT / "data" / "stego_suniward_04"
DATASETS = ("BOSSbase_1.01", "BOWS2")
MASTER_SEED = 20261119


def image_seed(dataset: str, index: int) -> int:
    key = f"SRNet:{MASTER_SEED}:{dataset}:{index}".encode("ascii")
    return (int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 2147483646) + 1


def check_pair(cover_path: Path, stego_path: Path) -> int:
    with Image.open(cover_path) as image:
        cover = np.asarray(image, dtype=np.int16)
    with Image.open(stego_path) as image:
        if image.mode != "L" or image.size != (256, 256):
            raise ValueError(f"Invalid stego image: {stego_path}")
        stego = np.asarray(image, dtype=np.int16)
    difference = stego - cover
    if np.max(np.abs(difference)) > 1:
        raise ValueError(f"Unexpected pixel change beyond +/-1: {stego_path}")
    changed = int(np.count_nonzero(difference))
    if changed == 0:
        raise ValueError(f"No embedding changes: {stego_path}")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--limit", type=int, help="Generate IDs 1..N per dataset for a trial")
    selection.add_argument("--all", action="store_true", help="Generate all 10,000 IDs per dataset")
    args = parser.parse_args()
    count = 10000 if args.all else args.limit
    if count is None or not 1 <= count <= 10000:
        parser.error("--limit must be between 1 and 10000")
    if not EXE.is_file():
        raise FileNotFoundError(EXE)

    summary = {}
    for dataset in DATASETS:
        source = SOURCE_ROOT / dataset
        target = TARGET_ROOT / dataset
        target.mkdir(parents=True, exist_ok=True)
        created = 0
        verified = 0
        changed_pixels = 0
        for index in range(1, count + 1):
            filename = f"{index}.pgm"
            cover = source / filename
            stego = target / filename
            if not cover.is_file():
                raise FileNotFoundError(cover)
            if not stego.exists():
                command = [str(EXE), "-i", str(cover), "-O", str(target),
                           "-a", "0.4", "-h", "0", "-r", str(image_seed(dataset, index))]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode or not stego.is_file():
                    raise RuntimeError(f"Embedding failed for {cover}: {result.stdout} {result.stderr}")
                created += 1
            else:
                verified += 1
            changed_pixels += check_pair(cover, stego)
            if index % 500 == 0 or index == count:
                print(f"{dataset}: {index}/{count} (created={created}, verified={verified})", flush=True)
        summary[dataset] = {"created": created, "verified": verified,
                            "mean_changed_pixels": round(changed_pixels / count, 2)}

    if args.all:
        TARGET_ROOT.mkdir(parents=True, exist_ok=True)
        with (TARGET_ROOT / "seeds.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("dataset", "image_id", "random_seed"))
            for dataset in DATASETS:
                for index in range(1, 10001):
                    writer.writerow((dataset, index, image_seed(dataset, index)))
        record = {
            "algorithm": "Official DDE Lab S-UNIWARD Windows C++ executable",
            "payload_bpp": 0.4,
            "stc_height": 0,
            "embedding": "optimal-coding simulation",
            "master_seed": MASTER_SEED,
            "executable_sha256": hashlib.sha256(EXE.read_bytes()).hexdigest(),
            "summary": summary,
        }
        (TARGET_ROOT / "embedding_manifest.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote manifest and seeds to {TARGET_ROOT}", flush=True)


if __name__ == "__main__":
    main()
