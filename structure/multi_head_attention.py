import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.q_linear(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        output, attn = self.scaled_dot_product_attention(Q, K, V, mask)  # Unpack tuple
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)
        return output, attn