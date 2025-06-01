import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 将输入的词汇表索引转换为指定维度的Embedding向量

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        """
        初始化TokenEmbedding
        :param vocab_size: 词汇表大小
        :param d_model: 嵌入向量的维度
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        """
        初始化位置编码
        :param d_model: 嵌入向量的维度
        :param max_len: 支持的最大序列长度
        :param device: 设备类型(CPU或GPU)
        """
        super(PositionalEncoding,self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim = 1)  # [max_len, 1]
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoding[:seq_len, :].to(x.device)

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, device):
        """
        初始化嵌入层和位置编码层
        :param vocab_size: 词汇表大小
        :param d_model: 嵌入向量的维度
        :param max_len: 支持的最大序列长度
        :param device: 设备类型(CPU或GPU)
        """
        super(Embedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size , d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, device)
        self.d_model = d_model

    def forward(self, x):
        """
        将输入的词汇表索引转换为嵌入向量，并添加位置编码
        :param x: 输入张量，形状为 [batch_size, seq_len]
        :return: 添加位置编码后的嵌入向量，形状为 [batch_size, seq_len, d_model]
        """
        token_embed = self.token_embedding(x)
        token_embed = token_embed * math.sqrt(self.d_model)  # 新增：缩放词嵌入
        pos_embed = self.positional_encoding(token_embed)
        return token_embed + pos_embed