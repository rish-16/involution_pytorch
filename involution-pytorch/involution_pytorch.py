import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange

'''
Unofficial implementation of the Involution operation
by Li et al. from CVPR 2021.

https://arxiv.org/abs/2103.06255
'''

class Inv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, red_ratio, padding, bn_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.red_ratio = red_ratio
        self.group_num = 0
        self.dilation = 2
        self.padding = padding

        self.out = nn.AvgPool2d() if self.stride > 1 else nn.Identity()
        self.reduce = nn.Conv2d(self.in_channels, self.in_channels // self.red_ratio, 1)
        self.span = nn.Conv2d(self.in_channels // self.red_ratio, self.kernel_size*self.kernel_size*self.group_num, 1)
        self.unfold = nn.Unfold(self.kernel_size, self.dilation, self.padding, self.stride)

        # dynamic kernel generation function
        bottleneck_dim = bn_dim
        self.kernel_net = nn.Sequential(
            nn.Linear(in_channels, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, kernel_size**2)
        )

    def forward(self, x):
        # implementation from the paper
        b, c, w, h = *x.shape
        x_unfolded = self.unfold(x)
        x_unfolded = x_unfolded.view()

        kernel = self.span(self.reduce(self.out(x)))
        kernel = kernel.view().unsqueeze(2)

        out = torch.mul(kernel, x_unfolded).sum(dim=3)
        out = out.view()

        return out