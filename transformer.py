import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, rate)
        self.out_layer = nn.Linear(d_model, 10)

    def forward(self, x, target, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(x, enc_padding_mask)  # [N,L_src,D]
        dec_output = self.decoder(
            target, enc_output, look_ahead_mask, dec_padding_mask)
        out = self.out_layer(dec_output)  # [N,L_tgt,D]
        return out


if __name__ == '__main__':
    m = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048)
    x = torch.randn(1, 50, 512)
    tgt = torch.randn(1, 25, 512)
    out = m(x, tgt, None, None, None)
    print(out.shape)
