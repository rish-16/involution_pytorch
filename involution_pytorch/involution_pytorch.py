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
    def __init__(self, channels, kernel_size, stride, group_ch=16, red_ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = channels
        self.out_channels = channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.red_ratio = red_ratio
        self.group_ch = group_ch
        self.groups = channels // self.group_ch
        self.dilation = 1
        self.padding = (kernel_size-1) // 2

        self.out = nn.AvgPool2d(stride, stride) if self.stride > 1 else nn.Identity()
        self.reduce = nn.Conv2d(channels, channels // self.red_ratio, kernel_size=1)
        self.span = nn.Conv2d(channels // self.red_ratio, kernel_size**2 * self.groups, kernel_size=1, stride=1)
        self.unfold = nn.Unfold(kernel_size, 1, self.padding, self.stride)

        # dynamic kernel generation function
        '''
        self.bottleneck_dim = 5
        self.kernel_net = nn.Sequential(
            nn.Linear(self.in_channels, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, kernel_size**2)
        )
        '''

    def forward(self, x):
        # implementation from the paper
        kernel = self.span(self.reduce(self.out(x)))
        b, c, h, w = kernel.shape
        kernel = kernel.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)

        x_unfolded = self.unfold(x)
        x_unfolded = x_unfolded.view(b, self.groups, self.group_ch, self.kernel_size**2, h, w)

        out = (kernel @ x_unfolded).sum(dim=3)
        out = out.view(b, self.out_channels, h, w)

        return out