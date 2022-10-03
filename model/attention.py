import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        """
        Args:
            embed_dim: the dimention of the model
            num_heads: the number of heads
            dropout: the dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        # 入力時の全結合層
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # 出力時の全結合層
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Args:
            query: (L, N, E)
            key: (S, N, E)
            value: (S, N, E)
            key_padding_mask: (N, S)
            attn_mask: (N * num_heads, L, S)
        """
        tgt_len, batch_size, embed_dim = query.shape

        # 入力時の変換
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        # scaled_dot_product
        output = self.scaled_dot_product(
            query, key, value, key_padding_mask, attn_mask)
        output = output.transpose(0, 1).contiguous().view(
            tgt_len, batch_size, embed_dim)
        output = self.out_linear(output)

        return output

    def scaled_dot_product(self, q, k, v, key_padding_mask, attn_mask):
        """
        Args:
            query: (L, N, E)
            key: (S, N, E)
            value: (S, N, E)
            key_padding_mask: (N, S)
            attn_mask: (N * num_heads, L, S)
        """
        tgt_len, batch_size, embed_dim = q.shape
        src_len, _, _ = k.shape

        # (L, N, E) -> (L, N * num_heads, E // num_heads) -> (N * num_heads, L, E // num_heads)
        q = q.contiguous().view(tgt_len, batch_size * self.num_heads,
                                self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, batch_size * self.num_heads,
                                self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, batch_size * self.num_heads,
                                self.head_dim).transpose(0, 1)

        # merge key_padding and attention mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len).expand(
                -1, self.num_heads, -1, -1).reshape(batch_size * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(
                    key_padding_mask, float('-inf'))

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill(attn_mask, float('-inf'))
            attn_mask = new_attn_mask

        # scale
        q = q / math.sqrt(embed_dim)

        # attn = (N * num_heads, L, S)
        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q, k.transpose(-2, -1))

        attn = F.softmax(attn, dim=-1)

        # (N * num_heads, L, E // num_heads)
        output = torch.bmm(attn, v)

        return output
