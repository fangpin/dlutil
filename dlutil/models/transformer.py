import torch
import torch.nn as nn
import math


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    logits = q @ k.transpose(-2, -1) / (d_k**0.5)
    if mask is not None:
        logits = logits.masked_fill(mask == 0, -1e9)
    attention = torch.nn.functional.softmax(logits, dim=-1)
    return attention @ v, attention


def expand_mask(mask):
    assert mask.ndim >= 2, (
        "Mask must be at least 2-dimensional with seq_length x seq_length"
    )
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MutiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, head):
        super().__init__()
        assert embed_dim % head == 0
        self.embed_dim = embed_dim
        self.head = head
        self.head_dim = embed_dim // head
        self.proj_qkv = nn.Linear(input_dim, 3 * embed_dim)
        self.proj_out = nn.Linear(embed_dim, input_dim)

    def forward(self, x, mask=None):
        batch, seq, _ = x.size()

        qkv = self.proj_qkv(x)
        qkv = qkv.reshape(batch, seq, self.head, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # batch, head, seq, feature

        q, k, v = qkv.chunk(3, dim=-1)
        if mask is not None:
            expand_mask(mask)
        o, attention = scaled_dot_product(q, k, v, mask)

        o = o.permute(0, 2, 1, 3)  # batch, seq, head, feature
        o = o.reshape(batch, seq, self.embed_dim)

        o = self.proj_out(o)
        return o, attention


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, forward_dim, dropout=0.0):
        super().__init__()
        self.mutil_head_att = MutiHeadAttention(input_dim, input_dim, num_heads)

        self.linear = nn.Sequential(
            nn.Linear(input_dim, forward_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(forward_dim, input_dim),
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x, mask=None):
        # attention part
        x = self.mutil_head_att(x, mask)
        x = x + self.dropout(x)
        x = self.norm1(x)

        # projection part
        z = self.linear(x)
        x = x + self.dropout(z)
        x = self.norm2(x)

        return x


class TrnasformerEncoder(nn.Module):
    def __init__(self, num_layers, input_dim, num_heads, forward_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(input_dim, num_heads, forward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq=5000):
        super().__init__()
        pe = torch.zeros(max_seq, d_model)
        position = torch.arange(0, max_seq, dtype=torch.float).unsqueeze(
            1
        )  # for broadcasting
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # add batch dimention
        self.register_buffer(
            "pe", pe, persistent=False
        )  # pin to device(e.g. GPU) memory

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]
