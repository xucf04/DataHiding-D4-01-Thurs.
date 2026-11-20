"""Durable CSV histories and report-ready SRNet figures."""

import csv
from pathlib import Path


TRAIN_COLUMNS = [
    "step", "utc_time", "learning_rate", "train_cross_entropy", "train_l2",
    "train_total_loss", "train_accuracy", "samples_seen", "interval_seconds",
    "steps_per_second", "gpu_peak_allocated_mib", "gpu_peak_reserved_mib",
]
VALID_COLUMNS = [
    "step", "utc_time", "valid_cross_entropy", "valid_l2",
    "valid_total_loss", "valid_accuracy", "valid_false_alarm_rate",
    "valid_miss_rate", "valid_pe_argmax", "valid_min_pe", "valid_auc",
    "valid_optimal_threshold", "valid_tn", "valid_fp", "valid_fn", "valid_tp",
    "evaluation_seconds", "best_checkpoint_selected",
]


class CsvHistory:
    def __init__(self, path: Path, columns: list[str], resume_step: int | None):
        self.path = path
        self.columns = columns
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if resume_step is None:
                raise FileExistsError(f"History exists without a checkpoint: {path}")
            with path.open("r", newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != columns:
                    raise ValueError(f"History columns do not match: {path}")
                rows = [row for row in reader if int(row["step"]) <= resume_step]
            with path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
        else:
            with path.open("w", newline="", encoding="utf-8") as target:
                csv.DictWriter(target, fieldnames=columns).writeheader()

    def append(self, row: dict):
        with self.path.open("a", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=self.columns)
            writer.writerow(row)


def _rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def plot_history(results_dir: Path):
    """Update PNG and SVG figures from the CSV histories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = _rows(results_dir / "train_history.csv")
    valid = _rows(results_dir / "validation_history.csv")
    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)

    def finish(name):
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        for extension in ("png", "svg"):
            plt.savefig(figures / f"{name}.{extension}", dpi=180)
        plt.close()

    if train:
        x = [int(row["step"]) for row in train]
        plt.figure(figsize=(8, 4.5))
        plt.plot(x, [float(row["train_cross_entropy"]) for row in train],
                 label="Train cross entropy (100-step mean)", linewidth=1.4)
        plt.plot(x, [float(row["train_total_loss"]) for row in train],
                 label="Train total loss (100-step mean)", linewidth=1.1)
        if valid:
            xv = [int(row["step"]) for row in valid]
            plt.plot(xv, [float(row["valid_cross_entropy"]) for row in valid],
                     "o-", label="Validation cross entropy", markersize=3)
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("SRNet training and validation loss")
        finish("loss_curve")

        plt.figure(figsize=(8, 4.5))
        plt.plot(x, [float(row["steps_per_second"]) for row in train],
                 label="Training throughput", linewidth=1.2)
        plt.xlabel("Training step")
        plt.ylabel("Steps per second")
        plt.title("SRNet training throughput")
        finish("throughput_curve")

    if valid:
        xv = [int(row["step"]) for row in valid]
        plt.figure(figsize=(8, 4.5))
        plt.plot(xv, [float(row["valid_accuracy"]) for row in valid],
                 "o-", label="Validation accuracy", markersize=3)
        plt.plot(xv, [float(row["valid_auc"]) for row in valid],
                 "o-", label="Validation AUC", markersize=3)
        plt.xlabel("Training step")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title("SRNet validation accuracy and AUC")
        finish("validation_scores")

        plt.figure(figsize=(8, 4.5))
        plt.plot(xv, [float(row["valid_pe_argmax"]) for row in valid],
                 "o-", label="P_e at argmax", markersize=3)
        plt.plot(xv, [float(row["valid_min_pe"]) for row in valid],
                 "o-", label="Minimum P_e", markersize=3)
        plt.xlabel("Training step")
        plt.ylabel("P_e")
        plt.ylim(0, 0.5)
        plt.title("SRNet validation detection error")
        finish("validation_pe")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate SRNet report figures from CSV histories")
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    arguments = parser.parse_args()
    plot_history(arguments.results_dir)
    print(f"Figures written to {arguments.results_dir / 'figures'}")
