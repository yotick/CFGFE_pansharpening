#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch.nn as nn


class CALayer(nn.Module):
    def __init__(self, channel: int, reduction: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x


class Refine(nn.Module):
    def __init__(self, n_feat: int, out_channel: int):
        super().__init__()
        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(CALayer(n_feat, 4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)
        return out

