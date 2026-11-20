"""Adamax update used by the official SRNet TensorFlow source."""

import torch


class OfficialAdamax(torch.optim.Optimizer):
    """Match SRNet's custom Adamax: no first-moment bias correction."""

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, beta1, beta2, eps = (group[key] for key in ("lr", "beta1", "beta2", "eps"))
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("OfficialAdamax does not support sparse gradients")
                state = self.state[param]
                if not state:
                    state["first_moment"] = torch.zeros_like(param)
                    state["infinity_norm"] = torch.zeros_like(param)
                first = state["first_moment"]
                infinity = state["infinity_norm"]
                first.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                torch.maximum(infinity.mul_(beta2).add_(eps), grad.abs(), out=infinity)
                param.addcdiv_(first, infinity, value=-lr)
        return loss
