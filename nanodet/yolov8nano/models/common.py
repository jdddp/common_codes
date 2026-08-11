import math

import torch
import torch.nn as nn


def make_divisible(value: float, divisor: int = 8) -> int:
    return int(math.ceil(value / divisor) * divisor)


def autopad(kernel_size: int, padding=None, dilation: int = 1) -> int:
    if padding is not None:
        return padding
    return ((kernel_size - 1) * dilation) // 2


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size),
            groups=groups,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConvBNAct(ConvBNAct):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        act: bool = True,
    ) -> None:
        groups = math.gcd(in_channels, out_channels)
        super().__init__(in_channels, out_channels, kernel_size, stride, groups=groups, act=act)


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True, expansion: float = 0.5) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.cv1 = ConvBNAct(channels, hidden, 1, 1)
        self.cv2 = ConvBNAct(hidden, channels, 3, 1)
        self.use_shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.use_shortcut else y


class C2f(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, repeats: int, shortcut: bool = False) -> None:
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden * 2, 1, 1)
        self.blocks = nn.ModuleList(Bottleneck(hidden, shortcut=shortcut, expansion=1.0) for _ in range(repeats))
        self.cv2 = ConvBNAct(hidden * (2 + repeats), out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        x1, x2 = x.chunk(2, dim=1)
        outputs = [x1, x2]
        for block in self.blocks:
            x2 = block(x2)
            outputs.append(x2)
        return self.cv2(torch.cat(outputs, dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_kernel: int = 5) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden, 1, 1)
        self.pool = nn.MaxPool2d(pool_kernel, stride=1, padding=pool_kernel // 2)
        self.cv2 = ConvBNAct(hidden * 4, out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))
