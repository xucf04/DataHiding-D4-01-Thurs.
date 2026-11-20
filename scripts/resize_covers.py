"""Resize the original BOSSbase and BOWS2 covers as in SRNet.

The vendored imresize.py implements MATLAB's default bicubic interpolation
with antialiasing. The bundled reference image was produced by MATLAB R2015b.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from imresize import imresize  # noqa: E402


def verify_matlab_fixture() -> None:
    fixture = ROOT / "vendor" / "matlab_imresize_test"
    with Image.open(fixture / "lena_512x512.png") as image:
        original = np.asarray(image)
    with Image.open(fixture / "lena_123x234_uint8.png") as image:
        expected = np.asarray(image)
    actual = imresize(original, output_shape=(123, 234))
    if not np.array_equal(actual, expected):
        raise RuntimeError("MATLAB R2015b reference fixture did not match")


def resize_one(source: Path, target: Path) -> str:
    with Image.open(source) as image:
        if image.size != (512, 512) or image.mode != "L":
            raise ValueError(f"Unexpected source image: {source} {image.size} {image.mode}")
        original = np.asarray(image)
    resized = imresize(original, scalar_scale=0.5)
    if resized.shape != (256, 256) or resized.dtype != np.uint8:
        raise RuntimeError(f"Unexpected resize result: {source}")
    if target.exists():
        with Image.open(target) as image:
            if image.mode != "L" or image.size != (256, 256):
                raise ValueError(f"Existing target has wrong format: {target}")
            if not np.array_equal(np.asarray(image), resized):
                raise ValueError(f"Existing target differs from expected pixels: {target}")
        return "verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    Image.fromarray(resized, mode="L").save(temporary, format="PPM")
    os.replace(temporary, target)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10000, help="IDs per source, for a trial run")
    args = parser.parse_args()
    if not 1 <= args.limit <= 10000:
        parser.error("--limit must be between 1 and 10000")
    verify_matlab_fixture()
    print("MATLAB R2015b reference fixture: exact pixel match", flush=True)

    summary = {}
    for name in ("BOSSbase_1.01", "BOWS2"):
        source_dir = ROOT / "data" / "raw" / name
        target_dir = ROOT / "data" / "cover_256" / name
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        actual_names = {path.name for path in source_dir.iterdir() if path.is_file()}
        expected_names = {f"{index}.pgm" for index in range(1, 10001)}
        if actual_names != expected_names:
            raise ValueError(f"Source file set differs from 1..10000: {source_dir}")
        counts = {"created": 0, "verified": 0}
        for index in range(1, args.limit + 1):
            filename = f"{index}.pgm"
            result = resize_one(source_dir / filename, target_dir / filename)
            counts[result] += 1
            if index % 1000 == 0 or index == args.limit:
                print(f"{name}: {index}/{args.limit} ({counts})", flush=True)
        summary[name] = counts

    if args.limit == 10000:
        source_code = ROOT / "vendor" / "imresize.py"
        record = {
            "method": "MATLAB imresize default bicubic with antialiasing, scale=0.5",
            "reference": "MATLAB R2015b fixture lena_123x234_uint8.png: exact pixel match",
            "source_code": "https://github.com/fatheral/matlab_imresize",
            "source_code_sha256": hashlib.sha256(source_code.read_bytes()).hexdigest(),
            "inputs": "512x512 uint8 grayscale PGM",
            "outputs": "256x256 uint8 grayscale PGM",
            "images_per_dataset": 10000,
            "summary": summary,
        }
        manifest = ROOT / "data" / "cover_256" / "resize_manifest.json"
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
