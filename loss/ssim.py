import math
import torch
from torch import nn
from torch.nn import functional as F

class SSIM3D(nn.Module):

    def __init__(self, window_size=11, size_average=True, channel=1):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        w = torch.Tensor([math.exp(-(i - window_size // 2) ** 2 / (2 * 1.5 ** 2)) for i in range(window_size)])
        self.register_buffer('window', w / w.sum())

    def gaussian_conv3d(self, x, window):
        k = window.to(x.device).type_as(x).repeat(self.channel, 1)
        n = self.window_size
        p = n // 2
        x = F.conv3d(x, k.view(self.channel, 1, n, 1, 1), padding=(p, 0, 0), groups=self.channel)
        x = F.conv3d(x, k.view(self.channel, 1, 1, n, 1), padding=(0, p, 0), groups=self.channel)
        return F.conv3d(x, k.view(self.channel, 1, 1, 1, n), padding=(0, 0, p), groups=self.channel)

    def forward(self, a, b, mask=None):
        a = a.clamp(0, 1)
        b = b.clamp(0, 1)
        conv = lambda x: self.gaussian_conv3d(x, self.window)
        ma, mb = (conv(a), conv(b))
        va = (conv(a * a) - ma.pow(2)).clamp_min(0)
        vb = (conv(b * b) - mb.pow(2)).clamp_min(0)
        cov = conv(a * b) - ma * mb
        value = (2 * ma * mb + 0.01 ** 2) * (2 * cov + 0.03 ** 2) / ((ma.pow(2) + mb.pow(2) + 0.01 ** 2) * (va + vb + 0.03 ** 2))
        value = torch.nan_to_num(value, nan=0.0)
        if mask is not None:
            valid = 1 - F.max_pool3d(1 - mask.float(), self.window_size, 1, self.window_size // 2)
            value = value * valid
            return value.sum() / (valid.sum() + 1e-08) if self.size_average else value
        return value.mean() if self.size_average else value

class SSIM_Loss(nn.Module):

    def __init__(self, window_size=11, size_average=True, do_3d=True):
        super().__init__()
        if not do_3d:
            raise ValueError('This release supports the 3D SSIM used by this method only')
        self.ssim = SSIM3D(window_size, size_average)

    def forward(self, a, b, mask=None):
        return 1 - self.ssim(a, b, mask)
