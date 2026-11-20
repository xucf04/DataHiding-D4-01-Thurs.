"""Create three ZIP deliverables and a non-data GitHub staging copy."""

from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "D4-01"
PACKAGE_NAMES = ("交付版代码.zip", "数据集和实验所用的图像.zip", "检查点.zip")


def all_files(directory: Path):
    return sorted((path for path in directory.rglob("*") if path.is_file()),
                  key=lambda path: path.relative_to(ROOT).as_posix())


def make_staging() -> None:
    if STAGING.exists():
        raise FileExistsError(f"Staging directory already exists: {STAGING}")
    existing = sorted(ROOT.iterdir(), key=lambda path: path.name)
    STAGING.mkdir()
    for source in existing:
        if source.name in {"data", "checkpoints", STAGING.name, *PACKAGE_NAMES}:
            continue
        target = STAGING / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        elif source.is_file():
            shutil.copy2(source, target)
        print(f"D4-01 copied: {source.name}", flush=True)

    (STAGING / ".gitignore").write_text(
        "# Data and trained weights stay in separate deliverables.\n"
        "data/\ncheckpoints/\n*.pt\n"
        "# Local caches and incomplete source checkout are not for GitHub.\n"
        "__pycache__/\n*.pyc\n_incomplete_git_clone/\n"
        "# Duplicated downloaded source archives.\n*.zip\n",
        encoding="utf-8",
    )
    (STAGING / "README.md").write_text(
        "# SRNet 复现实验（D4-01）\n\n"
        "运行方法见 [运行说明.md](运行说明.md)，实验过程、图表和最终指标见"
        "[实验步骤与结果索引.md](实验步骤与结果索引.md)。\n\n"
        "本目录是项目中除 `data/`、`checkpoints/` 和三个交付压缩包之外的文件副本。"
        "数据集和检查点分别保存在独立交付压缩包中。\n",
        encoding="utf-8",
    )


def code_files():
    names = (
        "train.py", "test.py", "paper_train.py", "paper_eval.py",
        "paper_metrics.py", "paper_optimizer.py", "paper_reporting.py",
        "report_visuals.py", "benchmark_speed.py", "README.md", ".gitignore",
    )
    files = [ROOT / "运行说明.md"]
    files.extend(ROOT / "srnet-reproduction" / name for name in names)
    files.extend(path for path in all_files(ROOT / "srnet-reproduction" / "model")
                 if "__pycache__" not in path.parts and path.suffix.lower() != ".pyc")
    files.extend(ROOT / "scripts" / name for name in ("resize_covers.py", "generate_stego.py"))
    files.extend(path for path in all_files(ROOT / "vendor")
                 if "__pycache__" not in path.parts and path.suffix.lower() not in (".zip", ".pyc"))
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def make_zip(output: Path, files: list[Path]) -> None:
    if output.exists():
        raise FileExistsError(f"Package already exists: {output}")
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary package already exists: {temporary}")
    source_bytes = sum(path.stat().st_size for path in files)
    print(f"Packaging {output.name}: {len(files)} files, {source_bytes / 2**30:.2f} GiB source", flush=True)
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for index, path in enumerate(files, start=1):
            suffix = path.suffix.lower()
            method = zipfile.ZIP_STORED if suffix in (".zip", ".gz", ".png", ".pt", ".jpg", ".jpeg") else zipfile.ZIP_DEFLATED
            archive.write(path, arcname=path.relative_to(ROOT).as_posix(),
                          compress_type=method, compresslevel=1 if method == zipfile.ZIP_DEFLATED else None)
            if index % 2000 == 0 or index == len(files):
                print(f"  {output.name}: {index}/{len(files)}", flush=True)
    temporary.replace(output)
    with zipfile.ZipFile(output) as archive:
        actual = archive.namelist()
        expected = [path.relative_to(ROOT).as_posix() for path in files]
        if actual != expected:
            raise RuntimeError(f"Package entries differ: {output}")
    print(f"Completed {output.name}: {output.stat().st_size / 2**30:.2f} GiB", flush=True)


def main() -> None:
    if any((ROOT / name).exists() for name in PACKAGE_NAMES):
        raise FileExistsError("One or more package files already exist; refusing to overwrite")
    if STAGING.exists():
        raise FileExistsError("D4-01 already exists; refusing to overwrite")
    make_staging()
    make_zip(ROOT / PACKAGE_NAMES[0], code_files())
    make_zip(ROOT / PACKAGE_NAMES[1], all_files(ROOT / "data"))
    make_zip(ROOT / PACKAGE_NAMES[2], all_files(ROOT / "checkpoints"))


if __name__ == "__main__":
    main()
