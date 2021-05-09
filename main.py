import torch
from involution_pytorch import Inv2D

inv = Inv2D()
x = torch.rand(1, 3, 256, 256)
y = inv(x)

print (y.shape)