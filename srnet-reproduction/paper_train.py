"""SRNet spatial S-UNIWARD 0.4 bpp training protocol (PyTorch).

Run from this directory with: python paper_train.py
Training requires all 20,000 stego images from scripts/generate_stego.py.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from model.model import Srnet
from paper_metrics import roc_summary
from paper_optimizer import OfficialAdamax
from paper_reporting import CsvHistory, TRAIN_COLUMNS, VALID_COLUMNS, plot_history


ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "boss_split_seed_20261119.json"
SPLIT_SEED = 20261119
PAIR_BATCH = 16
VALID_PAIR_BATCH = 20
TOTAL_STEPS = 500_000
LR_CHANGE_STEP = 400_000
VALID_EVERY = 5_000
TRAIN_LOG_EVERY = 100


def get_split() -> dict:
    if SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    else:
        ids = list(range(1, 10001))
        random.Random(SPLIT_SEED).shuffle(ids)
        split = {
            "seed": SPLIT_SEED,
            "boss_train": sorted(ids[:4000]),
            "boss_valid": sorted(ids[4000:5000]),
            "boss_test": sorted(ids[5000:]),
            "bows_train": list(range(1, 10001)),
        }
        SPLIT_FILE.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    train = set(split["boss_train"])
    valid = set(split["boss_valid"])
    test = set(split["boss_test"])
    if (len(train), len(valid), len(test)) != (4000, 1000, 5000):
        raise ValueError("Incorrect BOSSbase split sizes")
    if train & valid or train & test or valid & test or train | valid | test != set(range(1, 10001)):
        raise ValueError("BOSSbase split is overlapping or incomplete")
    if split["bows_train"] != list(range(1, 10001)):
        raise ValueError("BOWS2 training split is incomplete")
    return split


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, int]], augment: bool = False):
        self.pairs = pairs
        self.augment = augment
        self.cover_root = ROOT / "data" / "cover_256"
        self.stego_root = ROOT / "data" / "stego_suniward_04"

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        dataset, image_id = self.pairs[index]
        name = f"{image_id}.pgm"
        with Image.open(self.cover_root / dataset / name) as image:
            cover = np.asarray(image, dtype=np.float32)
        with Image.open(self.stego_root / dataset / name) as image:
            stego = np.asarray(image, dtype=np.float32)
        if cover.shape != (256, 256) or stego.shape != (256, 256):
            raise ValueError(f"Invalid pair dimensions: {dataset}/{name}")
        images = np.stack((cover, stego))
        if self.augment:
            images = np.rot90(images, random.randrange(4), axes=(1, 2))
            if random.randrange(2):
                images = np.flip(images, axis=2)
        return torch.from_numpy(np.ascontiguousarray(images[:, None]))


def initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        std = math.sqrt(1.3 * 2.0 / fan_in)
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std,
                              a=-2.0 * std, b=2.0 * std)
        nn.init.constant_(module.bias, 0.2)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.zeros_(module.bias)


@torch.no_grad()
def validation_metrics(model: nn.Module, loader: DataLoader, device: torch.device,
                       conv_weights, amp: bool, channels_last: bool) -> dict:
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.eval()
    ce_sum = torch.zeros((), device=device)
    all_scores = []
    all_predictions = []
    for pairs in loader:
        images = pairs.flatten(0, 1).to(
            device, non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format)
        labels = torch.arange(images.shape[0], device=device).remainder(2)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(images)
            ce = nn.functional.cross_entropy(logits, labels, reduction="sum")
        ce_sum += ce
        all_scores.append(logits.float().softmax(dim=1)[:, 1].cpu().numpy())
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())
    scores = np.concatenate(all_scores)
    predictions = np.concatenate(all_predictions)
    labels = np.arange(len(scores)) % 2
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    ce = float(ce_sum.item()) / len(scores)
    l2 = float((1e-4 * sum(weight.square().sum() for weight in conv_weights)).item())
    roc = roc_summary(scores)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    model.train()
    return {
        "valid_cross_entropy": ce,
        "valid_l2": l2,
        "valid_total_loss": ce + l2,
        "valid_accuracy": (tn + tp) / len(scores),
        "valid_false_alarm_rate": fp / (tn + fp),
        "valid_miss_rate": fn / (fn + tp),
        "valid_pe_argmax": 0.5 * (fp / (tn + fp) + fn / (fn + tp)),
        "valid_min_pe": roc["pe"],
        "valid_auc": roc["auc"],
        "valid_optimal_threshold": roc["threshold"],
        "valid_tn": tn, "valid_fp": fp, "valid_fn": fn, "valid_tp": tp,
        "evaluation_seconds": elapsed,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_run_config(path: Path, split: dict, workers: int,
                     precision: str, channels_last: bool, cudnn_benchmark: bool) -> None:
    source_files = ["paper_train.py", "paper_eval.py", "paper_metrics.py",
                    "paper_optimizer.py", "paper_reporting.py", "model/model.py", "model/utils.py"]
    source_hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in source_files
    }
    config = {
        "started_utc": utc_now(),
        "protocol": "SRNet spatial S-UNIWARD 0.4 bpp, BOSSbase+BOWS2",
        "split_seed": split["seed"],
        "split_manifest_sha256": hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest(),
        "boss_train_pairs": len(split["boss_train"]),
        "boss_valid_pairs": len(split["boss_valid"]),
        "boss_test_pairs": len(split["boss_test"]),
        "bows_train_pairs": len(split["bows_train"]),
        "pair_batch": PAIR_BATCH,
        "valid_pair_batch": VALID_PAIR_BATCH,
        "total_steps": TOTAL_STEPS,
        "lr_boundary_global_step": LR_CHANGE_STEP,
        "validation_interval": VALID_EVERY,
        "train_log_interval": TRAIN_LOG_EVERY,
        "workers": workers,
        "precision": precision,
        "channels_last": channels_last,
        "cudnn_benchmark": cudnn_benchmark,
        "optimizer": "OfficialAdamax",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
        "source_sha256": source_hashes,
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def save_checkpoint(path: Path, model: nn.Module, optimizer, step: int, accuracy: float,
                    precision: str, channels_last: bool) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "validation_accuracy": accuracy, "split_seed": SPLIT_SEED,
                "precision": precision, "channels_last": channels_last}, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="One synthetic GPU training step only")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Use FP32 instead of BF16 AMP")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--no-cudnn-benchmark", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    amp = not args.no_amp
    channels_last = not args.no_channels_last
    precision = "bf16" if amp else "fp32"
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 AMP is not supported on this GPU; use --no-amp")
    torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
    torch.manual_seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    random.seed(SPLIT_SEED)
    device = torch.device("cuda")
    model = Srnet().to(device)
    model.apply(initialize)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = OfficialAdamax(model.parameters(), lr=0.001)
    conv_weights = [module.weight for module in model.modules() if isinstance(module, nn.Conv2d)]

    if args.smoke:
        images = torch.rand(PAIR_BATCH * 2, 1, 256, 256, device=device) * 255
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = torch.arange(PAIR_BATCH * 2, device=device).remainder(2)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = nn.functional.cross_entropy(model(images), labels)
        loss = loss + 1e-4 * sum(weight.square().sum() for weight in conv_weights)
        loss.backward()
        optimizer.step()
        print(f"Smoke step passed: loss={loss.item():.4f}; batch={len(images)}; "
              f"precision={precision}; channels_last={channels_last}")
        return

    split = get_split()
    train_pairs = [("BOSSbase_1.01", i) for i in split["boss_train"]]
    train_pairs += [("BOWS2", i) for i in split["bows_train"]]
    valid_pairs = [("BOSSbase_1.01", i) for i in split["boss_valid"]]
    for dataset, image_id in train_pairs + valid_pairs:
        filename = f"{image_id}.pgm"
        for parent in (ROOT / "data" / "cover_256", ROOT / "data" / "stego_suniward_04"):
            if not (parent / dataset / filename).is_file():
                raise FileNotFoundError(parent / dataset / filename)
    train_loader = DataLoader(PairDataset(train_pairs, augment=True), batch_size=PAIR_BATCH,
                              shuffle=True, drop_last=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0)
    valid_loader = DataLoader(PairDataset(valid_pairs), batch_size=VALID_PAIR_BATCH,
                              shuffle=False, num_workers=args.workers, pin_memory=True)
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    last_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best_valid_final_100k.pt"
    start_step = 0
    best_accuracy = -1.0
    if args.resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("precision", "fp32") != precision or state.get("channels_last", False) != channels_last:
            raise ValueError("Checkpoint precision or memory format differs from this run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        if best_path.exists():
            best_accuracy = torch.load(best_path, map_location="cpu", weights_only=False)["validation_accuracy"]
    elif last_path.exists():
        raise FileExistsError("Checkpoint exists: use --resume or move it aside")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    config_path = results_dir / "run_config.json"
    if not config_path.exists():
        write_run_config(config_path, split, args.workers, precision, channels_last,
                         torch.backends.cudnn.benchmark)
    train_history = CsvHistory(results_dir / "train_history.csv", TRAIN_COLUMNS,
                               start_step if args.resume else None)
    valid_history = CsvHistory(results_dir / "validation_history.csv", VALID_COLUMNS,
                               start_step if args.resume else None)
    writer = SummaryWriter(log_dir=str(results_dir / "tensorboard"),
                           purge_step=start_step + 1 if args.resume else None,
                           flush_secs=30)
    print(f"train={len(train_pairs)} pairs valid={len(valid_pairs)} pairs "
          f"start={start_step} precision={precision} channels_last={channels_last}", flush=True)
    iterator = iter(train_loader)
    model.train()
    torch.cuda.synchronize()
    window_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    window_ce = torch.zeros((), device=device)
    window_l2 = torch.zeros((), device=device)
    window_total = torch.zeros((), device=device)
    window_correct = torch.zeros((), device=device)
    progress = tqdm(range(start_step + 1, TOTAL_STEPS + 1), total=TOTAL_STEPS,
                    initial=start_step, unit="step", dynamic_ncols=True,
                    mininterval=1.0)
    for step in progress:
        try:
            pairs = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            pairs = next(iterator)
        images = pairs.flatten(0, 1).to(
            device, non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format)
        labels = torch.arange(images.shape[0], device=device).remainder(2)
        # TF evaluates piecewise_constant before global_step is incremented.
        lr = 0.001 if step - 1 <= LR_CHANGE_STEP else 0.0001
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(images)
            ce = nn.functional.cross_entropy(logits, labels)
        l2 = 1e-4 * sum(weight.square().sum() for weight in conv_weights)
        loss = ce + l2
        loss.backward()
        optimizer.step()
        window_ce += ce.detach()
        window_l2 += l2.detach()
        window_total += loss.detach()
        window_correct += (logits.argmax(dim=1) == labels).sum().detach()
        if step % TRAIN_LOG_EVERY == 0:
            torch.cuda.synchronize()
            interval_seconds = time.perf_counter() - window_start
            train_row = {
                "step": step,
                "utc_time": utc_now(),
                "learning_rate": lr,
                "train_cross_entropy": window_ce.item() / TRAIN_LOG_EVERY,
                "train_l2": window_l2.item() / TRAIN_LOG_EVERY,
                "train_total_loss": window_total.item() / TRAIN_LOG_EVERY,
                "train_accuracy": window_correct.item() / (TRAIN_LOG_EVERY * PAIR_BATCH * 2),
                "samples_seen": step * PAIR_BATCH * 2,
                "interval_seconds": interval_seconds,
                "steps_per_second": TRAIN_LOG_EVERY / interval_seconds,
                "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            }
            train_history.append(train_row)
            for name in ("cross_entropy", "l2", "total_loss", "accuracy"):
                writer.add_scalar(f"train/{name}", train_row[f"train_{name}"], step)
            writer.add_scalar("train/learning_rate", lr, step)
            writer.add_scalar("performance/steps_per_second", train_row["steps_per_second"], step)
            writer.add_scalar("performance/gpu_peak_allocated_mib",
                              train_row["gpu_peak_allocated_mib"], step)
            progress.set_postfix_str(
                f"CE={train_row['train_cross_entropy']:.4f} "
                f"loss={train_row['train_total_loss']:.4f} "
                f"acc={train_row['train_accuracy']:.3f} "
                f"speed={train_row['steps_per_second']:.2f}/s")
            window_ce.zero_()
            window_l2.zero_()
            window_total.zero_()
            window_correct.zero_()
            window_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
        if step % VALID_EVERY == 0:
            metrics = validation_metrics(model, valid_loader, device, conv_weights,
                                         amp, channels_last)
            accuracy = metrics["valid_accuracy"]
            selected = step > LR_CHANGE_STEP and accuracy > best_accuracy
            metrics.update(step=step, utc_time=utc_now(),
                           best_checkpoint_selected=int(selected))
            valid_history.append(metrics)
            for name in ("cross_entropy", "l2", "total_loss", "accuracy",
                         "false_alarm_rate", "miss_rate", "pe_argmax", "min_pe", "auc"):
                writer.add_scalar(f"validation/{name}", metrics[f"valid_{name}"], step)
            writer.flush()
            tqdm.write(f"step={step} valid_loss={metrics['valid_total_loss']:.5f} "
                       f"valid_acc={accuracy:.4f} valid_min_pe={metrics['valid_min_pe']:.4f} "
                       f"valid_auc={metrics['valid_auc']:.4f}")
            save_checkpoint(last_path, model, optimizer, step, accuracy,
                            precision, channels_last)
            if selected:
                best_accuracy = accuracy
                save_checkpoint(best_path, model, optimizer, step, accuracy,
                                precision, channels_last)
            try:
                plot_history(results_dir)
            except Exception as exc:
                print(f"Warning: could not update report figures: {exc}", flush=True)
            # Keep validation, checkpoint I/O, and plotting out of training throughput.
            torch.cuda.synchronize()
            window_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
    writer.close()


if __name__ == "__main__":
    main()
