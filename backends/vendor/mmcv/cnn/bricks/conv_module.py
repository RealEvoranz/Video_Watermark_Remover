"""Simplified ConvModule used by E2FGVI SPyNet."""

from __future__ import annotations

import torch.nn as nn


class ConvModule(nn.Module):
    """Conv + optional activation block compatible with E2FGVI SPyNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool | str = "auto",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        inplace: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        del conv_cfg, kwargs

        if norm_cfg is not None:
            raise NotImplementedError("norm_cfg is not supported in mmcv compat layer")

        use_bias = True if bias == "auto" else bool(bias)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )

        self.with_activation = act_cfg is not None
        if self.with_activation:
            act_type = act_cfg.get("type", "ReLU")
            if act_type == "ReLU":
                self.activate = nn.ReLU(inplace=inplace)
            elif act_type == "LeakyReLU":
                self.activate = nn.LeakyReLU(
                    negative_slope=act_cfg.get("negative_slope", 0.1),
                    inplace=inplace,
                )
            else:
                raise NotImplementedError(f"Unsupported activation: {act_type}")
        else:
            self.activate = None

    def forward(self, x, activate: bool = True, norm: bool = True) -> nn.Tensor:
        del norm
        x = self.conv(x)
        if activate and self.with_activation and self.activate is not None:
            x = self.activate(x)
        return x
