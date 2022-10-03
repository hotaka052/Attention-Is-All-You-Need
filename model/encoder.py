from torch import nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .utils import _get_clones


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm):
        """
        Args:
            encoder_layer: instance of the EncoderLayer
            num_layers: the number of sub-layers in the encoder
            norm: the layer normalization component
        """
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layer = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Args:
            src: the sequence to the encoder (S, N, E)
            src_mask: mask for src (N * num_heads, S, S)
            src_key_padding_mask: (N, S)
        """
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)

        output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff=2048, dropout=0.1):
        """
        Args:
            d_model: the number of expected features in the encoder/decoder inputs
            nhead: the number of heads in the multiheadattention models
            dim_ff: the dimension of the feedforward network model
            dropout: the dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()

        # for Self-attention block
        self.self_attention = MultiHeadAttention(
            d_model, nhead, dropout=dropout)
        self.dropout_sa = nn.Dropout(dropout)

        # for FeedForward block
        self.ff = FeedForward(d_model, dim_ff=dim_ff, dropout=dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args
            src: the sequence to the encoder (S, N, E)
            src_mask: mask for src (N * num_heads, S, S)
            src_key_padding_mask: (N, S)
        """
        x = src
        x = self.norm_1(x + self._self_attention_block(x,
                        src_mask, src_key_padding_mask))
        x = self.norm_2(x + self._ff_block(x))

        return x

    def _self_attention_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attention(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        return self.dropout_sa(x)

    def _ff_block(self, x):
        x = self.ff(x)

        return self.dropout_ff(x)
