"""Loss functions for from-scratch image restoration training.

All losses here are implemented without any pretrained weights or external data
to comply with the homework rules:
    - No external data
    - No pretrained weights
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


def frequency_loss(pred, target):
    fp = torch.fft.rfft2(pred, norm="ortho")
    ft = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(
        torch.stack([fp.real, fp.imag], dim=-1),
        torch.stack([ft.real, ft.imag], dim=-1),
    )


def _gaussian_window(window_size: int, sigma: float):
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        g = _gaussian_window(window_size, sigma)
        kernel_2d = g[:, None] * g[None, :]
        self.register_buffer("kernel", kernel_2d.view(1, 1, window_size, window_size))
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    def _filter(self, x):
        c = x.shape[1]
        kernel = self.kernel.expand(c, 1, self.window_size, self.window_size)
        return F.conv2d(x, kernel, padding=self.window_size // 2, groups=c)

    def forward(self, pred, target):
        mu_p = self._filter(pred)
        mu_t = self._filter(target)
        mu_pp = mu_p * mu_p
        mu_tt = mu_t * mu_t
        mu_pt = mu_p * mu_t
        sigma_pp = self._filter(pred * pred) - mu_pp
        sigma_tt = self._filter(target * target) - mu_tt
        sigma_pt = self._filter(pred * target) - mu_pt
        ssim = ((2 * mu_pt + self.C1) * (2 * sigma_pt + self.C2)) / (
            (mu_pp + mu_tt + self.C1) * (sigma_pp + sigma_tt + self.C2)
        )
        return 1.0 - ssim.mean()
