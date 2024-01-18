
import torch
import torch.nn as nn


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, c1, c2, dimension=1):
        super().__init__()
        self.d = dimension
        self.cv = nn.Conv2d(4 * c1, c2, 1, 1)

    def forward(self, x):
        x_out = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.cv(x_out)
