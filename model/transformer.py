from torch import nn

from .decoder import TransformerDecoder, TransformerDecoderLayer
from .encoder import TransformerEncoder, TransformerEncoderLayer


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_ff=2048, dropout=0.1):
        """
        Args:
            d_model: the number of expected features in the encoder/decoder inputs
            nhead: the number of heads in the multiheadattention models
            num_layers: the number of sub-layers in the encoder and decoder
            dim_ff: the dimension of the feedforward network model
            dropout: the dropout rate
        """

        super(Transformer, self).__init__()

        # encoderの準備
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer, num_layers, encoder_norm)

        # decoderの準備
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            src: the sequence to the encoder (S, N, E)
            tgt: the sequence to the decoder (T, N, E)
            src_mask: mask for src (N * num_heads, S, S)
            tgt_mask: mask for target (N * num_heads, T, T)
            memory_mask: mask for memory (T, S)
            src_key_padding_mask: (N, S)
            tgt_key_padding_mask: (N, T)
            memory_key_padding_mask: (N, S)
        """

        memory = self.encoder(src, mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
