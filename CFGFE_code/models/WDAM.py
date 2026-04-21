#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from torch.autograd import Function


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = F.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = F.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = F.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = F.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = F.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = F.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave: str):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave: str):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer("w_ll", w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_lh", w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hl", w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hh", w_hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class WDAM(nn.Module):
    """
    Wavelet-based Dual-domain Attention Module (WDAM).

    This file is extracted as-is from the original framework and kept self-contained
    for the standalone CFGFE release.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: int = 2,
        alpha: float = 1,
        sr_ratio: int = 1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)

        self.dim = dim
        self.w_heads = int(num_heads * alpha)
        self.w_dim = self.w_heads * head_dim
        self.s_heads = num_heads - self.w_heads
        self.s_dim = self.s_heads * head_dim
        self.ws = window_size

        if self.ws == 1:
            self.s_heads = 0
            self.s_dim = 0
            self.w_heads = num_heads
            self.w_dim = dim

        self.scale = qk_scale or head_dim**-0.5

        if self.w_heads > 0:
            self.dwt = DWT_2D(wave="haar")
            self.idwt = IDWT_2D(wave="haar")

            self.reduce = nn.Sequential(
                nn.Conv2d(self.dim, self.w_dim // 4, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(self.w_dim // 4, affine=True),
                nn.ReLU(inplace=True),
            )

            self.filter = nn.Sequential(
                nn.Conv2d(self.w_dim, self.w_dim, kernel_size=3, padding=1, stride=1, groups=1),
                nn.BatchNorm2d(self.w_dim, affine=True),
                nn.ReLU(inplace=True),
            )

            self.w_kv_embed = (
                nn.Conv2d(self.w_dim, self.w_dim, kernel_size=sr_ratio, stride=sr_ratio)
                if sr_ratio > 1
                else nn.Identity()
            )
            self.w_q = nn.Linear(self.dim, self.w_dim, bias=qkv_bias)
            self.w_kv = nn.Sequential(nn.LayerNorm(self.w_dim), nn.Linear(self.w_dim, self.w_dim * 2))
            self.w_proj = nn.Linear(self.w_dim + self.w_dim // 4, self.w_dim)

        if self.s_heads > 0:
            self.s_qkv = nn.Linear(self.dim, self.s_dim * 3, bias=qkv_bias)
            self.s_proj = nn.Linear(self.s_dim, self.s_dim)

    def sa(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = (
            self.s_qkv(x)
            .reshape(B, total_groups, -1, 3, self.s_heads, self.s_dim // self.s_heads)
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.s_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.s_dim)
        x = self.s_proj(x)
        return x

    def wa(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        q = self.w_q(x).reshape(B, H * W, self.w_heads, self.w_dim // self.w_heads).permute(0, 2, 1, 3)
        x2 = x.permute(0, 3, 1, 2)
        x_dwt = self.dwt(self.reduce(x2))
        x_dwt = self.filter(x_dwt)

        x_idwt = self.idwt(x_dwt)
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-2) * x_idwt.size(-1)).transpose(1, 2)

        kv = self.w_kv_embed(x_dwt).reshape(B, self.w_dim, -1).permute(0, 2, 1)
        kv = (
            self.w_kv(kv)
            .reshape(B, -1, 2, self.w_heads, self.w_dim // self.w_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x3 = (attn @ v).transpose(1, 2).reshape(B, H * W, self.w_dim)
        x3 = self.w_proj(torch.cat([x3, x_idwt], dim=-1))
        x3 = x3.reshape(B, H, W, self.w_dim)
        return x3

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)

        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        if self.s_heads == 0:
            out = self.wa(x)
            if pad_r > 0 or pad_b > 0:
                out = out[:, :H, :W, :]
            return out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        if self.w_heads == 0:
            out = self.sa(x)
            if pad_r > 0 or pad_b > 0:
                out = out[:, :H, :W, :]
            return out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Both branches exist; keep the original behavior used in the integrated framework.
        sa_out = self.sa(x)
        _wa_out = self.wa(x)
        out = sa_out
        out = out.reshape(B, N, C)
        return out

