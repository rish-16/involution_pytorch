import torch
from involution_pytorch import Inv2d

inv = Inv2d(
    channels=16,
    kernel_size=3,
    stride=1
)

x = torch.rand(1, 16, 32, 32)
y = inv(x) # (1, 16, 32, 32)