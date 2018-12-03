import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, layers, padding_idx):
        super(Encoder, self).__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_size, layers)

    def forward(self, inputs, h_0, c_0, lengths):
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, lengths=lengths.numpy())
        return self.lstm(packed, (h_0, c_0))

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, layers, padding_idx):
        super(Decoder, self).__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_size, layers)

    def forward(self, inputs, h_0):
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        return self.lstm(embedded, (h_0, ))