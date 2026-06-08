"""Modulated deformable convolution via torchvision.ops.deform_conv2d."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


def _as_int_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return _pair(value)


def modulated_deform_conv2d(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    deform_groups: int = 1,
) -> torch.Tensor:
    """mmcv-compatible wrapper around torchvision deform_conv2d (DCNv2)."""
    del groups, deform_groups

    stride = _as_int_pair(stride)
    padding = _as_int_pair(padding)
    dilation = _as_int_pair(dilation)

    return deform_conv2d(
        input,
        offset,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=mask,
    )


class ModulatedDeformConv2d(nn.Module):
    """Drop-in replacement for mmcv.ops.ModulatedDeformConv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.init_weights()

    def init_weights(self) -> None:
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )
