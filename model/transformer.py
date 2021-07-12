import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .stem import VGGStem, ResNetStem
from .pos_encoding import LearnedPosEncoding


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, dropout)
        self.linear = nn.Linear(d_model, 10)

    def forward(self, x, target, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_out = self.encoder(x, enc_padding_mask)  # [N,L_src,D]
        dec_out = self.decoder(
            target, enc_out, look_ahead_mask, dec_padding_mask)
        out = self.linear(dec_out)  # [N,L_tgt,D]
        return out


class ConViT(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.stem = ResNetStem(d_model)
        self.pos_encoding = LearnedPosEncoding(8*8+1, d_model, dropout)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, dropout)
        self.linear = nn.Linear(d_model, 10)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        N = x.size(0)
        out = self.stem(x)
        out = out.reshape(N, self.d_model, -1)  # [N,D,L]
        out = out.permute(0, 2, 1)  # [N,L,D]
        cls_tokens = self.cls_token.expand(N, -1, -1)
        out = torch.cat([cls_tokens, out], dim=1)
        out = self.pos_encoding(out)
        out = self.encoder(out, None)
        out = out[:, 0, :]
        out = self.linear(out.view(N, -1))
        return out


class SEViT(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.stem = VGGStem(d_model)
        self.pos_encoding = LearnedPosEncoding(4*4, d_model, dropout)

        #  self.encoder = Encoder(num_layers, d_model, num_heads, dff, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.w = nn.Linear(d_model, 1)
        self.linear = nn.Linear(d_model, 10)

    def forward(self, x):
        N = x.size(0)
        out = self.stem(x)
        out = out.reshape(N, self.d_model, -1)  # [N,D,L]
        out = out.permute(0, 2, 1)  # [N,L,D]
        out = self.pos_encoding(out)
        out = self.encoder(out, None)
        w = self.w(out).softmax(dim=-2)
        out = out.transpose(1, 2) @ w
        out = self.linear(out.view(N, -1))
        return out


def test_transformer():
    m = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048)
    x = torch.randn(1, 50, 512)
    tgt = torch.randn(1, 25, 512)
    out = m(x, tgt, None, None, None)
    print(out.shape)


def test_vit():
    m = ConViT(num_layers=2, d_model=256, num_heads=8, dff=1024)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(y.shape)


if __name__ == '__main__':
    test_vit()
