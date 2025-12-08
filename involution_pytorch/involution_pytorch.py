import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Unofficial implementation of the Involution operation
by Li et al. from CVPR 2021.

https://arxiv.org/abs/2103.06255
'''

class Inv2d(nn.Module):
    def __init__(self, channels, kernel_size, stride, group_ch=16, red_ratio=2, **kwargs):
        # Call parent constructor
        super().__init__(**kwargs)
        
        # Store core configuration
        self.in_channels = int(channels)
        self.out_channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.red_ratio = int(red_ratio)
        self.group_ch = int(group_ch)
        
        # Compute number of groups and validate divisibility
        if self.in_channels % self.group_ch != 0:
            raise ValueError(
                f"Inv2d: in_channels ({self.in_channels}) must be divisible by "
                f"group_ch ({self.group_ch})."
            )
        self.groups = self.in_channels // self.group_ch
        
        # Set dilation and padding used for unfold
        self.dilation = 1
        self.padding = (self.kernel_size - 1) // 2
        
        # Define optional pooling for kernel generation when stride > 1
        self.out = nn.AvgPool2d(self.stride, self.stride) if self.stride > 1 else nn.Identity()
        
        # Define reduction convolution used in kernel generator
        self.reduce = nn.Conv2d(
            self.in_channels,
            self.in_channels // self.red_ratio,
            kernel_size=1
        )
        
        # Define span convolution producing k^2 * groups channels
        self.span = nn.Conv2d(
            self.in_channels // self.red_ratio,
            self.kernel_size ** 2 * self.groups,
            kernel_size=1,
            stride=1
        )
        
        # Define unfold operator to extract k x k patches
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )

    def forward(self, x):
        # Unpack input shape and validate channel count
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"Inv2d: expected input with {self.in_channels} channels, got {c}."
            )
        
        # Generate kernels using pooled features
        pooled = self.out(x)
        kernel = self.span(self.reduce(pooled))
        
        # Unpack kernel shape and compute derived quantities
        b_k, c_k, h_out, w_out = kernel.shape
        if b_k != b:
            raise RuntimeError(
                f"Inv2d: batch mismatch between input ({b}) and kernel ({b_k})."
            )
        
        # Compute k^2 from configured kernel size
        k2 = self.kernel_size ** 2
        
        # Validate kernel channel dimension
        expected_kernel_channels = self.groups * k2
        if c_k != expected_kernel_channels:
            raise RuntimeError(
                f"Inv2d: expected kernel channels {expected_kernel_channels}, got {c_k}."
            )
        
        # Reshape kernel to (B, groups, K, H_out, W_out)
        kernel = kernel.view(
            b,
            self.groups,
            k2,
            h_out,
            w_out
        )
        
        # Extract local patches with unfold
        patches = self.unfold(x)
        
        # Unpack unfolded patches shape
        b_u, c_u, l = patches.shape
        if b_u != b:
            raise RuntimeError(
                f"Inv2d: batch mismatch between input ({b}) and patches ({b_u})."
            )
        
        # Validate spatial length of unfolded patches
        expected_l = h_out * w_out
        if l != expected_l:
            raise RuntimeError(
                f"Inv2d: spatial mismatch between kernel ({h_out}x{w_out}) "
                f"and unfolded patches length ({l})."
            )
        
        # Validate unfolded channel dimension
        expected_unfold_channels = self.in_channels * k2
        if c_u != expected_unfold_channels:
            raise RuntimeError(
                f"Inv2d: expected unfolded channels {expected_unfold_channels}, got {c_u}."
            )
        
        # Reshape patches to (B, groups, group_ch, K, H_out, W_out)
        patches = patches.view(
            b,
            self.groups,
            self.group_ch,
            k2,
            h_out,
            w_out
        )
        
        # Contract over K using einsum to get (B, groups, group_ch, H_out, W_out)
        out = torch.einsum(
            "bgkij,bgckij->bgcij",
            kernel,
            patches
        )
        
        # Reshape to (B, C, H_out, W_out)
        out = out.reshape(
            b,
            self.out_channels,
            h_out,
            w_out
        )
        
        # Return involution output
        return out
