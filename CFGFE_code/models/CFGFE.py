#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pywt  # type: ignore
from torch.autograd import Function

from .refine import Refine
from .WDAM import WDAM


def initialize_weights(net_l, scale: float = 1.0) -> None:
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


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


class KernelNorm(nn.Module):
    def __init__(self, in_channels: int, filter_type: str):
        super().__init__()
        assert filter_type in ("spatial", "spectral")
        assert in_channels >= 1
        self.in_channels = in_channels
        self.filter_type = filter_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.filter_type == "spatial":
            b, _, h, w = x.size()
            x = x.reshape(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
            x = x.reshape(b, _, h, w)
        elif self.filter_type == "spectral":
            b = x.size(0)
            c = self.in_channels
            x = x.reshape(b, c, -1)
            x = x - x.mean(dim=2).reshape(b, c, 1)
            x = x / (x.std(dim=2).reshape(b, c, 1) + 1e-10)
        else:
            raise RuntimeError(f"Unsupported filter type {self.filter_type}")
        return x


class KernelGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size_list=(1, 3, 5),
        stride: int = 1,
        padding_list=(0, 1, 2),
        se_ratio: float = 0.5,
    ):
        super().__init__()
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.spatial_branch = nn.ModuleList()
        self.spectral_branch = nn.ModuleList()
        self.in_channels = in_channels
        assert se_ratio > 0
        mid_channels = int(in_channels * se_ratio)

        self.cross_spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(in_channels, 2, 1, 1, 0),
            nn.Sigmoid(),
        )

        for i in range(len(kernel_size_list)):
            kernel_size, padding = kernel_size_list[i], padding_list[i]
            spatial_kg = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=in_channels,
                ),
                nn.Conv2d(in_channels=in_channels, out_channels=kernel_size**2, kernel_size=1),
                nn.Conv2d(
                    in_channels=kernel_size**2,
                    out_channels=kernel_size**2,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=kernel_size**2,
                ),
                nn.Conv2d(in_channels=kernel_size**2, out_channels=kernel_size**2, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    in_channels=kernel_size**2,
                    out_channels=kernel_size**2,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=kernel_size**2,
                ),
                nn.Conv2d(in_channels=kernel_size**2, out_channels=kernel_size**2, kernel_size=1),
            )
            spectral_kg = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=mid_channels, out_channels=in_channels * kernel_size**2, kernel_size=1),
            )
            self.spatial_branch.append(spatial_kg)
            self.spectral_branch.append(spectral_kg)
        self.spatial_norm = KernelNorm(in_channels=1, filter_type="spatial")
        self.spectral_norm = KernelNorm(in_channels=in_channels, filter_type="spectral")

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        b, c, h, w = x.shape
        attn = self.cross_spatial_attn(torch.cat([x, y], 1))
        attn1, attn2 = torch.chunk(attn, 2, 1)
        x, y = x * attn1, y * attn2

        outputs, spatial_kernels, spectral_kernels = [], [], []
        for i, k in enumerate(self.kernel_size_list):
            spatial_kernel = self.spatial_branch[i](y)  # [b, 1*k**2, h, w]
            spectral_kernel = self.spectral_branch[i](x)  # [b, c*k**2, 1, 1]
            spectral_kernel = spectral_kernel.reshape(b, self.in_channels, k**2)
            spatial_kernels.append(spatial_kernel)
            spectral_kernels.append(spectral_kernel)
        k_square = [k**2 for k in self.kernel_size_list]
        spatial_kernels = torch.cat(spatial_kernels, dim=1)
        spatial_kernels = self.spatial_norm(spatial_kernels)
        spatial_kernels = spatial_kernels.split(k_square, dim=1)
        spectral_kernels = torch.cat(spectral_kernels, dim=-1)
        spectral_kernels = self.spectral_norm(spectral_kernels)
        spectral_kernels = spectral_kernels.split(k_square, dim=-1)

        for i, k in enumerate(self.kernel_size_list):
            spatial_kernel = spatial_kernels[i].permute(0, 2, 3, 1).reshape(b, 1, h, w, k, k)
            spectral_kernel = spectral_kernels[i].reshape(b, c, 1, 1, k, k)
            self.adaptive_kernel = torch.mul(spectral_kernel, spatial_kernel)
            output = self.adaptive_conv(x, i)
            outputs.append(output)
        return outputs

    def adaptive_conv(self, x: torch.Tensor, i: int) -> torch.Tensor:
        b, c, h, w = x.shape
        pad = self.padding_list[i]
        k = self.kernel_size_list[i]
        kernel = self.adaptive_kernel
        if pad > 0:
            x_pad = torch.zeros(b, c, h + 2 * pad, w + 2 * pad, device=x.device, dtype=x.dtype)
            x_pad[:, :, pad:-pad, pad:-pad] = x
        else:
            x_pad = x
        x_pad = F.unfold(x_pad, (k, k))
        x_pad = x_pad.reshape(b, c, k, k, h, w).permute(0, 1, 4, 5, 2, 3)
        return torch.sum(torch.mul(x_pad, kernel), [4, 5])


class SAFM(nn.Module):
    def __init__(self, in_channels: int, channels: int, if_proj: bool = False):
        super().__init__()
        self.kpn = KernelGenerator(channels)
        self.if_proj = if_proj
        if if_proj:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 5, 1, 2),
            )
        self.out_conv = nn.Conv2d(in_channels=channels * 3, out_channels=channels, kernel_size=3, stride=1, padding=1)

    def forward(self, lms: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        if self.if_proj:
            lms = self.proj(lms)
        f1, f3, f5 = self.kpn(lms, pan)
        out = self.out_conv(torch.cat([f1, f3, f5], dim=1))
        return out


class FDMM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")
        self.safm = SAFM(channels, channels)
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, msf: torch.Tensor, panf: torch.Tensor) -> torch.Tensor:
        pan_dwt = self.dwt(self.pre1(panf))
        ms_dwt = self.dwt(self.pre2(msf))
        B, C, H, W = pan_dwt.shape
        qc = C // 4
        pan_ll, pan_lh, pan_hl, pan_hh = (
            pan_dwt[:, 0:qc, :, :],
            pan_dwt[:, qc : 2 * qc, :, :],
            pan_dwt[:, 2 * qc : 3 * qc, :, :],
            pan_dwt[:, 3 * qc : 4 * qc, :, :],
        )
        ms_ll, ms_lh, ms_hl, ms_hh = (
            ms_dwt[:, 0:qc, :, :],
            ms_dwt[:, qc : 2 * qc, :, :],
            ms_dwt[:, 2 * qc : 3 * qc, :, :],
            ms_dwt[:, 3 * qc : 4 * qc, :, :],
        )

        ll = self.safm(pan_ll, ms_ll)
        lh = self.safm(pan_lh, ms_lh)
        hl = self.safm(pan_hl, ms_hl)
        hh = self.safm(pan_hh, ms_hh)
        x = torch.cat([ll, lh, hl, hh], dim=1)
        out = self.idwt(x)
        return self.post(out)


class DFCE(nn.Module):
    def __init__(self, in_channel: int, out_factor: int):
        super().__init__()
        self.align_layer = nn.Conv2d(in_channel // 2, in_channel, 3, 1, 1, bias=True)
        self.conv_down_a = nn.Conv2d(in_channel, in_channel, 3, 2, 1, bias=True)
        self.conv_up_a = nn.ConvTranspose2d(in_channel, in_channel, 3, 2, 1, 1, bias=True)
        self.conv_down_b = nn.Conv2d(in_channel, in_channel, 3, 2, 1, bias=True)
        self.conv_up_b = nn.ConvTranspose2d(in_channel, in_channel, 3, 2, 1, 1, bias=True)
        self.conv_cat = nn.Conv2d(in_channel * 2, in_channel * out_factor, 3, 1, 1, bias=True)
        self.active = nn.ReLU(inplace=True)

    def forward(self, source_feature: torch.Tensor, enhanced_feature: torch.Tensor) -> torch.Tensor:
        source_feature = self.conv_down_a(self.align_layer(source_feature))

        res_a = self.active(self.conv_down_a(enhanced_feature)) - source_feature
        out_a = self.active(self.conv_up_a(res_a)) + enhanced_feature

        res_b = source_feature - self.active(self.conv_down_b(enhanced_feature))
        out_b = self.active(self.conv_up_b(res_b + source_feature))
        out = self.active(self.conv_cat(torch.cat([out_a, out_b], dim=1)))
        return out


class SpaFre(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.panprocess = nn.Conv2d(channels, channels, 3, 1, 1)
        self.panpre = nn.Conv2d(channels, channels, 1, 1, 0)

        self.coarse_process = WDAM(
            dim=64,
            num_heads=8,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            window_size=1,
            alpha=1,
            sr_ratio=1,
        )
        self.fine_process = FDMM(channels)

        self.conv_f = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv_f1 = nn.Conv2d(32, 16, 3, 1, 1)
        self.dfce = DFCE(in_channel=32, out_factor=2)
        self.conv_out = nn.Conv2d(64, channels, 3, 1, 1)

    def forward(self, msf: torch.Tensor, pan: torch.Tensor):
        panpre = self.panprocess(pan)
        panf = self.panpre(panpre)

        x = torch.cat([msf, panf], 1)
        B, H, W, C = x.shape
        x1 = x.reshape(B, H * W, C)
        coarse_fuse = self.coarse_process(x1, H, W)
        fine_fuse = self.fine_process(msf, panf)
        coarse_fuse = coarse_fuse.permute(0, 2, 1, 3)
        coarse_fuse = self.conv_f1(coarse_fuse)
        drf = self.dfce(coarse_fuse, self.conv_f(fine_fuse))

        drf_out = self.conv_out(drf)
        out = drf_out + msf
        return out, panpre


class FeatureProcess(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_p = nn.Conv2d(4, channels, 3, 1, 1)
        self.conv_p1 = nn.Conv2d(1, channels, 3, 1, 1)
        self.block = SpaFre(channels)
        self.block1 = SpaFre(channels)
        self.block2 = SpaFre(channels)
        self.block3 = SpaFre(channels)
        self.block4 = SpaFre(channels)
        self.fuse = nn.Conv2d(5 * channels, channels, 1, 1, 0)

    def forward(self, ms: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        msf = self.conv_p(ms)
        panf = self.conv_p1(pan)
        msf0, panf0 = self.block(msf, panf)
        msf1, panf1 = self.block1(msf0, panf0)
        msf2, panf2 = self.block2(msf1, panf1)
        msf3, panf3 = self.block3(msf2, panf2)
        msf4, panf4 = self.block4(msf3, panf3)
        msout = self.fuse(torch.cat([msf0, msf1, msf2, msf3, msf4], 1))
        return msout


class CFGFE(nn.Module):
    def __init__(self, num_channels: int = 16):
        super().__init__()
        self.process = FeatureProcess(num_channels)
        self.refine = Refine(num_channels, 4)

    def forward(self, l_ms: torch.Tensor, bms: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        if pan is None:
            raise ValueError("pan must be provided")

        mHR = bms
        HRf = self.process(mHR, pan)
        HR = self.refine(HRf) + mHR
        return HR

