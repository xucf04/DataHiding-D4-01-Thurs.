"""Evaluate the fixed BOSSbase test split with the paper SRNet protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.model import Srnet
from paper_metrics import confusion_at, roc_summary
from paper_train import ROOT, PairDataset, get_split


@torch.no_grad()
def scores_for(model, pairs, batch_size, workers, device, amp, channels_last):
    loader = DataLoader(PairDataset(pairs), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=True)
    scores = []
    for batch in loader:
        images = batch.flatten(0, 1).to(
            device, non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(images)
        scores.append(logits.float().softmax(dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "best_valid_final_100k.pt")
    parser.add_argument("--batch-pairs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    split = get_split()
    if state["split_seed"] != split["seed"]:
        raise ValueError("Checkpoint and split manifest disagree")
    device = torch.device("cuda")
    model = Srnet().to(device)
    model.load_state_dict(state["model"])
    precision = state.get("precision", "fp32")
    channels_last = state.get("channels_last", False)
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"Unsupported checkpoint precision: {precision}")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    valid_pairs = [("BOSSbase_1.01", i) for i in split["boss_valid"]]
    test_pairs = [("BOSSbase_1.01", i) for i in split["boss_test"]]
    valid_scores = scores_for(model, valid_pairs, args.batch_pairs, args.workers,
                              device, precision == "bf16", channels_last)
    valid_roc = roc_summary(valid_scores)
    test_scores = scores_for(model, test_pairs, args.batch_pairs, args.workers,
                             device, precision == "bf16", channels_last)
    test_roc = roc_summary(test_scores)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": state["step"],
        "checkpoint_precision": precision,
        "checkpoint_channels_last": channels_last,
        "split_seed": split["seed"],
        "user_acceptance_pe_limit": 0.18184,
        "paper_table_ii_suni_train_suni_test_pe": 0.1023,
        "meets_user_acceptance_pe_limit": bool(test_roc["pe"] <= 0.18184),
        "confusion_layout": "rows=true [cover,stego], columns=predicted [cover,stego]",
        "validation": {"pairs": len(valid_pairs), **valid_roc},
        "test": {
            "pairs": len(test_pairs),
            "paper_min_pe": test_roc["pe"],
            "auc": test_roc["auc"],
            "confusion_at_test_optimal_threshold": confusion_at(test_scores, test_roc["threshold"]),
            "test_optimal_threshold": test_roc["threshold"],
            "confusion_at_validation_threshold": confusion_at(test_scores, valid_roc["threshold"]),
        },
    }
    output = ROOT / "results" / "evaluation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["test"], indent=2))
    print(f"Assigned P_e <= 0.18184: {'PASS' if result['meets_user_acceptance_pe_limit'] else 'FAIL'}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
