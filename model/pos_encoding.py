import torch
import torch.nn as nn
import torch.nn.functional as F


class PosEncoding(nn.Module):
    '''
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    '''

    def __init__(self):
        super().__init__()

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / (10000. ** (2. * (i//2) / d_model))
        return pos * angle_rates

    def forward(self, x):
        N, L, D = x.shape
        angle_rads = self.get_angles(
            torch.arange(L, device=x.device)[:, None],
            torch.arange(D, device=x.device)[None, :],
            D
        )
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = angle_rads[:, 0::2].sin()
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = angle_rads[:, 1::2].cos()
        return x + angle_rads[None, :, :]


if __name__ == '__main__':
    m = PosEncoding()
    x = torch.randn(1, 10, 512)
    y = m(x)
    print(y.shape)
