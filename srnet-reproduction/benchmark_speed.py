"""Short GPU-only SRNet training benchmark; does not touch datasets or checkpoints."""

import gc
import time

import torch
from torch import nn

from model.model import Srnet
from paper_optimizer import OfficialAdamax
from paper_train import PAIR_BATCH, initialize


def run(name, amp, channels_last, cudnn_benchmark, fp16=False, warmup=5, measured=15):
    torch.backends.cudnn.benchmark = cudnn_benchmark
    model = Srnet().cuda()
    model.apply(initialize)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = OfficialAdamax(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    weights = [module.weight for module in model.modules() if isinstance(module, nn.Conv2d)]
    images = torch.rand(PAIR_BATCH * 2, 1, 256, 256, device="cuda") * 255
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    labels = torch.arange(len(images), device="cuda") % 2
    torch.cuda.reset_peak_memory_stats()
    started = None
    for step in range(warmup + measured):
        if step == warmup:
            torch.cuda.synchronize()
            started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16 if fp16 else torch.bfloat16,
                            enabled=amp):
            logits = model(images)
            ce = nn.functional.cross_entropy(logits, labels)
        loss = ce + 1e-4 * sum(weight.square().sum() for weight in weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(f"{name}: {measured / elapsed:.2f} steps/s; "
          f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB; "
          f"loss {loss.item():.4f}", flush=True)
    del model, optimizer, scaler, weights, images, labels, logits, ce, loss
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run("FP32", False, False, False)
    run("BF16 AMP", True, False, False)
    run("BF16 AMP + channels_last", True, True, False)
    run("BF16 AMP + channels_last + cudnn benchmark", True, True, True)
    run("FP16 AMP + channels_last + cudnn benchmark", True, True, True, fp16=True)
