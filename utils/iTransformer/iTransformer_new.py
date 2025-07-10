import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Inverted Transformer model for time series forecasting.
    Paper link: https://arxiv.org/abs/2310.06625

    Args:
        seq_len (int): Input sequence length
        pred_len (int): Prediction sequence length
        d_model (int): Dimension of model
        embed (str): Type of embedding
        freq (str): Frequency for time features
        dropout (float): Dropout rate
        factor (int): Factor for attention
        n_heads (int): Number of attention heads
        d_ff (int): Dimension of feedforward network
        e_layers (int): Number of encoder layers
        activation (str): Activation function
        output_attention (bool): Whether to output attention weights
        use_norm (bool): Whether to use normalization
        class_strategy (str): Classification strategy (if applicable)
    """

    def __init__(self,
                 seq_len=96,
                 pred_len=96,
                 d_model=128,
                 embed='timeF',
                 freq='d',
                 dropout=0.1,
                 factor=1,
                 n_heads=8,
                 d_ff=128,
                 e_layers=2,
                 activation='gelu',
                 output_attention=False,
                 use_norm=False,
                 class_strategy=None):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.use_norm = use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        self.class_strategy = class_strategy

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]