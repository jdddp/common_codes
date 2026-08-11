from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000.0) -> None:
        self.ema = deepcopy(self._unwrap(model)).eval()
        self.decay = decay
        self.tau = tau
        self.updates = 0
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def _unwrap(self, model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def _decay(self) -> float:
        return self.decay * (1 - torch.exp(torch.tensor(-self.updates / self.tau)).item())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self._decay()
        msd = self._unwrap(model).state_dict()
        for key, value in self.ema.state_dict().items():
            if not value.dtype.is_floating_point:
                value.copy_(msd[key])
                continue
            value.mul_(decay).add_(msd[key].detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def update_attr(self, model: nn.Module) -> None:
        for key, value in self._unwrap(model).__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (int, float, str, tuple, list, dict)):
                setattr(self.ema, key, value)
