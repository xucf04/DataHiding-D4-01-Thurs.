"""Create report-ready ROC, confusion, resize, and embedding figures.

Run from the project root: python .\srnet-reproduction\report_visuals.py
The saved test scores are reused on later runs; --recompute-scores reruns inference.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model.model import Srnet
from paper_eval import scores_for
from paper_metrics import confusion_at, roc_summary
from paper_train import ROOT, get_split


RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SCORES = RESULTS / "test_scores.csv"
EVALUATION = RESULTS / "evaluation.json"


def save_figure(fig, basename: str) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = FIGURES / f"{basename}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path.relative_to(ROOT).as_posix())
    plt.close(fig)
    return paths


def test_scores(split: dict, evaluation: dict, recompute: bool, workers: int,
                batch_pairs: int) -> np.ndarray:
    ids = split["boss_test"]
    checkpoint = ROOT / "checkpoints" / "best_valid_final_100k.pt"
    if SCORES.exists() and not recompute:
        with SCORES.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != 2 * len(ids):
            raise ValueError("Cached test score count differs from fixed test split")
        for index, (image_id, label) in enumerate((i, y) for i in ids for y in (0, 1)):
            row = rows[index]
            if (int(row["image_id"]), int(row["label"]), int(row["checkpoint_step"])) != (
                image_id, label, evaluation["checkpoint_step"]
            ):
                raise ValueError(f"Cached test score order differs at row {index + 1}")
        scores = np.array([float(row["stego_probability"]) for row in rows], dtype=np.float32)
        print(f"Reused {SCORES.relative_to(ROOT)}", flush=True)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to create test scores; reuse an existing CSV instead")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["step"] != evaluation["checkpoint_step"] or state["split_seed"] != split["seed"]:
            raise ValueError("Checkpoint, split, and evaluation metadata disagree")
        precision = state.get("precision", "fp32")
        if precision not in ("fp32", "bf16"):
            raise ValueError(f"Unsupported checkpoint precision: {precision}")
        channels_last = state.get("channels_last", False)
        model = Srnet().cuda()
        model.load_state_dict(state["model"])
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        model.eval()
        pairs = [("BOSSbase_1.01", image_id) for image_id in ids]
        scores = scores_for(model, pairs, batch_pairs, workers, torch.device("cuda"),
                            precision == "bf16", channels_last)
        print(f"Inferred {len(scores)} held-out test images", flush=True)

    summary = roc_summary(scores)
    expected = evaluation["test"]
    if abs(summary["pe"] - expected["paper_min_pe"]) > 1e-6 or abs(summary["auc"] - expected["auc"]) > 1e-6:
        raise ValueError(f"Test scores do not reproduce evaluation.json: {summary}")
    if confusion_at(scores, expected["test_optimal_threshold"])["matrix"] != expected[
        "confusion_at_test_optimal_threshold"
    ]["matrix"]:
        raise ValueError("Test scores do not reproduce the recorded confusion matrix")

    if not SCORES.exists() or recompute:
        temporary = SCORES.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(("dataset", "image_id", "label", "stego_probability", "checkpoint_step"))
            for position, score in enumerate(scores):
                writer.writerow(("BOSSbase_1.01", ids[position // 2], position % 2,
                                 repr(float(score)), evaluation["checkpoint_step"]))
        temporary.replace(SCORES)
        print(f"Saved {SCORES.relative_to(ROOT)}", flush=True)
    return scores


def roc_points(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.arange(len(scores)) % 2
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tpr = np.r_[0, np.cumsum(sorted_labels == 1)[ends]] / int((labels == 1).sum())
    fpr = np.r_[0, np.cumsum(sorted_labels == 0)[ends]] / int((labels == 0).sum())
    return fpr, tpr


def plot_roc(scores: np.ndarray, evaluation: dict) -> list[str]:
    fpr, tpr = roc_points(scores)
    test = evaluation["test"]
    matrix = test["confusion_at_test_optimal_threshold"]["matrix"]
    fp = matrix[0][1]
    tp = matrix[1][1]
    fig, ax = plt.subplots(figsize=(6.2, 5.7))
    ax.plot(fpr, tpr, color="#1766a1", linewidth=2.0,
            label=f"SRNet, AUC = {test['auc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1,
            label="Chance")
    ax.scatter(fp / 5000, tp / 5000, color="#d95f02", zorder=4,
               label=f"Minimum $P_e$ = {test['paper_min_pe']:.2%}")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Held-out BOSSbase test ROC (S-UNIWARD 0.4 bpp)",
           xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    return save_figure(fig, "test_roc")


def plot_confusion(matrix: list[list[int]], title: str, basename: str,
                   pe: float) -> list[str]:
    values = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5.8, 5.1))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=5000)
    fig.colorbar(image, ax=ax, label="Number of images", fraction=0.046, pad=0.04)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Cover", "Stego"], yticklabels=["Cover", "Stego"],
           xlabel="Predicted class", ylabel="True class",
           title=f"{title}\n$P_e$ = {pe:.2%}")
    for row in range(2):
        for col in range(2):
            number = int(values[row, col])
            ax.text(col, row, f"{number:,}\n({number / 5000:.1%})",
                    ha="center", va="center", fontsize=13,
                    color="white" if number > 2500 else "#14283e")
    fig.tight_layout()
    return save_figure(fig, basename)


def plot_examples(split: dict) -> tuple[list[str], list[str], int, int]:
    image_id = split["boss_train"][0]
    filename = f"{image_id}.pgm"
    with Image.open(ROOT / "data" / "raw" / "BOSSbase_1.01" / filename) as image:
        raw = np.asarray(image, dtype=np.uint8)
    with Image.open(ROOT / "data" / "cover_256" / "BOSSbase_1.01" / filename) as image:
        cover = np.asarray(image, dtype=np.uint8)
    with Image.open(ROOT / "data" / "stego_suniward_04" / "BOSSbase_1.01" / filename) as image:
        stego = np.asarray(image, dtype=np.uint8)
    if raw.shape != (512, 512) or cover.shape != (256, 256) or stego.shape != (256, 256):
        raise ValueError("Illustration source images have unexpected dimensions")
    difference = stego.astype(np.int16) - cover.astype(np.int16)
    if np.max(np.abs(difference)) > 1:
        raise ValueError("Stego illustration includes a change beyond +/-1")
    changed = int(np.count_nonzero(difference))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, array, title in zip(axes, (raw, cover),
                                ("Original cover: 512 x 512", "MATLAB-style resize: 256 x 256")):
        ax.imshow(array, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"BOSSbase training image #{image_id}")
    fig.tight_layout()
    resize_files = save_figure(fig, "resize_example")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    for ax, array, title, cmap, vmax in zip(
        axes, (cover, stego, np.abs(difference)),
        ("Cover", "S-UNIWARD stego (0.4 bpp)", "Changed pixels (absolute difference)"),
        ("gray", "gray", "magma"), (255, 255, 1)
    ):
        ax.imshow(array, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"BOSSbase training pair #{image_id}: {changed:,} / 65,536 pixels changed ({changed / 65536:.2%})")
    fig.tight_layout()
    embedding_files = save_figure(fig, "embedding_example")
    return resize_files, embedding_files, image_id, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-scores", action="store_true",
                        help="Run held-out test inference even if test_scores.csv exists")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-pairs", type=int, default=20)
    args = parser.parse_args()
    split = get_split()
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    if evaluation["split_seed"] != split["seed"] or evaluation["test"]["pairs"] != len(split["boss_test"]):
        raise ValueError("Evaluation and fixed split disagree")
    scores = test_scores(split, evaluation, args.recompute_scores, args.workers,
                         args.batch_pairs)
    roc_files = plot_roc(scores, evaluation)
    optimal = evaluation["test"]["confusion_at_test_optimal_threshold"]
    fixed = evaluation["test"]["confusion_at_validation_threshold"]
    optimal_files = plot_confusion(optimal["matrix"], "Held-out test: ROC-optimal threshold",
                                   "confusion_test_optimal", optimal["pe"])
    fixed_files = plot_confusion(fixed["matrix"], "Held-out test: validation-fixed threshold",
                                 "confusion_validation_threshold", fixed["pe"])
    resize_files, embedding_files, example_id, changed = plot_examples(split)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_script": "srnet-reproduction/report_visuals.py",
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": "checkpoints/best_valid_final_100k.pt",
        "checkpoint_step": evaluation["checkpoint_step"],
        "test_score_file": SCORES.relative_to(ROOT).as_posix(),
        "test_images": len(scores),
        "test_auc": roc_summary(scores)["auc"],
        "test_min_pe": roc_summary(scores)["pe"],
        "example_dataset": "BOSSbase_1.01",
        "example_split": "boss_train",
        "example_image_id": example_id,
        "example_changed_pixels": changed,
        "figures": {
            "test_roc": roc_files,
            "confusion_test_optimal": optimal_files,
            "confusion_validation_threshold": fixed_files,
            "resize_example": resize_files,
            "embedding_example": embedding_files,
        },
    }
    output = RESULTS / "visualization_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output.relative_to(ROOT)} and {sum(map(len, manifest['figures'].values()))} figures", flush=True)


if __name__ == "__main__":
    main()
