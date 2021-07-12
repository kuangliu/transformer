import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .encoder import point_wise_feed_forward_network


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        # enc_output: [N,L_src,D]
        attn1, attn_w1 = self.mha1(x, x, x, look_ahead_mask)  # [N,L_tgt,D]
        attn1 = F.dropout(attn1, p=self.dropout)
        out1 = self.norm1(attn1 + x)

        attn2, attn_w2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = F.dropout(attn2, p=self.dropout)
        out2 = self.norm1(attn2 + out1)  # [N,L,D]

        ffn_out = self.ffn(out2)  # [N,L,D]
        ffn_out = F.dropout(ffn_out, p=self.dropout)
        out = self.norm3(ffn_out + out2)  # [N,l,D]
        return out, attn_w1, attn_w2


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout)
                                     for _ in range(num_layers)])

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        for layer in self.layers:
            out, _, _ = layer(out, enc_output, look_ahead_mask, padding_mask)
        return out


def test_decoder_layer():
    m = DecoderLayer(512, 8, 2048)
    x = torch.randn(64, 50, 512)
    enc_out = torch.randn(64, 43, 512)
    out, _, _ = m(x, enc_out, None, None)
    print(out.shape)


def test_decoder():
    m = Decoder(num_layers=2, d_model=512, num_heads=8, dff=2048)
    L = 25
    x = torch.randn(64, L, 512)
    enc_output = torch.randn(64, L, 512)
    y = m(x, enc_output, None, None)
    print(y.shape)


if __name__ == '__main__':
    test_decoder_layer()
