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


class ViT(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = self.make_tokenizer()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.linear = nn.Linear(d_model, 10)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def make_tokenizer(self):
        layers = [
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, self.d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        N = x.size(0)
        out = self.tokenizer(x)
        out = out.reshape(N, self.d_model, -1)  # [N,D,L]
        out = out.permute(0, 2, 1)  # [N,L,D]

        cls_tokens = self.cls_token.expand(N, -1, -1)
        out = torch.cat([cls_tokens, out], dim=1)

        out = self.encoder(out, None)
        out = out[:, 0, :]
        out = self.linear(out.view(N, -1))
        return out


def test_transformer():
    m = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048)
    x = torch.randn(1, 50, 512)
    tgt = torch.randn(1, 25, 512)
    out = m(x, tgt, None, None, None)
    print(out.shape)


def test_vit():
    m = ViT(num_layers=2, d_model=256, num_heads=8, dff=1024)
    x = torch.randn(1, 3, 32, 32)
    y = m(x)
    print(y.shape)


if __name__ == '__main__':
    test_vit()
