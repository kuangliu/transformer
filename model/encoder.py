import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention


def point_wise_feed_forward_network(d_model, dff):
  return nn.Sequential(
      nn.Linear(d_model, dff),  # [N,L,dff]
      nn.ReLU(True),
      nn.Linear(dff, d_model),  # [N,L,D]
  )


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_out, _ = self.mha(x, x, x, mask)  # [N,L,D]
        attn_out = F.dropout(attn_out, p=self.dropout)
        attn_out = self.norm1(x + attn_out)    # [N,L,D]

        ffn_out = self.ffn(attn_out)           # [N,L,D]
        ffn_out = F.dropout(ffn_out, p=self.dropout)
        out = self.norm2(attn_out + ffn_out)   # [N,L,D]
        return out


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout)
                                     for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


def test_encoder_layer():
    m = EncoderLayer(d_model=512, num_heads=8, dff=128)
    x = torch.randn(64, 43, 512)  # [N,L,D]
    y = m(x, mask=None)  # [N,L,D]
    print(y.shape)


def test_encoder():
    m = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048)
    x = torch.randn(64, 20, 512)
    y = m(x, mask=None)
    print(y.shape)


if __name__ == '__main__':
    test_encoder()
