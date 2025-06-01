import torch
import torch.nn as nn
from structure.multi_head_attention import MultiHeadAttention
from structure.feed_forward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # encoder_layer.py
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask=mask)  # 统一为 mask
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x