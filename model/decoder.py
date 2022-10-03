from torch import nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .utils import _get_clones


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        """
        Args:
            decoder_layer: instance of the EncoderLayer
            num_layers: the number of sub-layers in the decoder
            norm: the layer normalization component
        """
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        """
        Args:
            tgt: the sequence to the decoder (T, N, E)
            memory: the sequence from the last layer of the encoder
            tgt_mask: mask for target (N * num_heads, T, T)
            memory_mask: mask for memory (T, S)
            tgt_key_padding_mask: (N, T)
            memory_key_padding_mask: (N, S)
        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff=2048, dropout=0.1):
        """
        Args:
            d_model: the number of expected features in the encoder/decoder inputs
            nhead: the number of heads in the multiheadattention models
            dim_ff: the dimension of the feedforward network model
            dropout: the dropout rate
        """
        super(TransformerDecoderLayer, self).__init__()

        # for Self-attention block
        self.self_attention = MultiHeadAttention(
            d_model, nhead, dropout=dropout)
        self.dropout_sa = nn.Dropout(dropout)

        # for Source Traget-attention block
        self.source_target_attention = MultiHeadAttention(
            d_model, nhead, dropout=dropout)
        self.dropout_sta = nn.Dropout(dropout)

        # for FeedForward block
        self.ff = FeedForward(d_model, dim_ff=dim_ff, dropout=dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: the sequence to the decoder (T, N, E)
            memory: the sequence from the last layer of the encoder
            tgt_mask: mask for target (N * num_heads, T, T)
            memory_mask: mask for memory (T, S)
            tgt_key_padding_mask: (N, T)
            memory_key_padding_mask: (N, S)
        """
        x = tgt
        x = self.norm_1(x + self._self_attention_block(x,
                        tgt_mask, tgt_key_padding_mask))
        x = self.norm_2(x + self._source_target_attention_block(x,
                        memory, memory_mask, memory_key_padding_mask))
        x = self.norm_3(x + self._ff_block(x))

        return x

    def _self_attention_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attention(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        return self.dropout_sa(x)

    def _source_target_attention_block(self, x, memory, attn_mask, key_padding_mask):
        x = self.source_target_attention(
            x, memory, memory, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        return self.dropout_sta(x)

    def _ff_block(self, x):
        x = self.ff(x)

        return self.dropout_ff(x)
