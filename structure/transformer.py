import torch
import torch.nn as nn
from structure.decoder_layer import DecoderLayer
from structure.encoder_layer import  EncoderLayer
from structure.embedding import Embedding


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=5000,device='cuda',dropout=0.1):
        super(Transformer, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.encoder_embedding = Embedding(src_vocab_size, d_model, max_len, device)
        self.decoder_embedding = Embedding(tgt_vocab_size, d_model, max_len, device)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 1).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 1).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=tgt.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.encoder_embedding(src))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        tgt_embedded = self.dropout(self.decoder_embedding(tgt))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output


if __name__ == "__main__":
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    batch_size = 64
    seq_length = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, device=device)
    model.to(device)

    src = torch.randint(1, src_vocab_size, (batch_size, seq_length), device=device)
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, seq_length), device=device)

    output = model(src, tgt)
    print(output.shape)  # Expected: [64, 20, 10000]