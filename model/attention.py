import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert(self.d_model % self.num_heads == 0)

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        '''Split the last dimension into (num_heads, depth).

        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        '''
        N, L, D = x.shape
        x = x.view(N, L, self.num_heads, -1)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, q, k, v, mask):
        '''
        Attention(Q,K,V) = softmax(QK/sqrt(d)) * V

        The mask is multiplied with -1e9 (close to negative infinity). 
        This is done because the mask is summed with the scaled 
        matrix multiplication of Q and K and is applied immediately before a softmax. 
        The goal is to zero out these cells, and large negative inputs to softmax are near zero in the output.        

        Args:
          q: query shape (N,num_heads,Lq,depth)
          k: key shape (N,num_heads,Lk,depth)
          v: value shape (N,num_heads,Lv,depth)
          mask: mask shape (N,num_heads,Lq,Lk)

        Returns:
          out: sized [N,M,Lq,depth]
          attention_weights: sized [N,M,Lq,Lk]
        '''
        N, M, Lq, D = q.shape
        q = q.reshape(N*M, -1, D)
        k = q.reshape(N*M, -1, D)
        v = q.reshape(N*M, -1, D)

        qk = q @ k.transpose(1, 2)  # [N*M,Lq,Lk]
        scaled_qk = qk * (D ** -0.5)

        # Add mask to scaled qk.
        if mask is not None:
            mask = mask.view(N*M, Lq, -1)
            scaled_qk += (mask * -1e9)

        attention_weights = F.softmax(scaled_qk, dim=-1)  # [N*M,Lq,Lk]
        out = attention_weights @ v  # [N*M,Lq,depth]
        return out.view(N, M, Lq, D), attention_weights.view(N, M, Lq, -1)

    def forward(self, q, k, v, mask):
        N = q.size(0)

        q = self.wq(q)  # [N,Lq,D]
        k = self.wk(k)  # [N,Lk,D]
        v = self.wv(v)  # [N,Lv,D]

        # Lq = Lk = Lv = L
        q = self.split_heads(q)  # [N,num_heads,Lq,depth]
        k = self.split_heads(k)  # [N,num_heads,Lk,depth]
        v = self.split_heads(v)  # [N,num_heads,Lv,depth]

        # scaled_attention:  [N,num_heads,Lq,depth]
        # attention_weights: [N,num_heads,Lq,Lk]
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        # [N,Lq,num_heads,depth]
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        # [N,Lq,d_model]
        scaled_attention = scaled_attention.reshape(N, -1, self.d_model)

        out = self.linear(scaled_attention)
        return out, attention_weights


if __name__ == '__main__':
    m = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(1, 60, 512)  # [N,L,D]
    out, attn = m(q=x, k=x, v=x, mask=None)
    print(out.shape)
    print(attn.shape)
