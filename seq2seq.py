import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, layers, dev, pad):
        super(Encoder, self).__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.pad = pad
        self.dev = dev

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=layers, batch_first=True)

    def forward(self, inputs, lengths):
        i, l = inputs.to(self.dev), lengths.to(self.dev)
        embedded = self.embedding(i)
        out = pack_padded_sequence(embedded, l, batch_first=True)
        return self.gru(out)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, layers, dev, pad, sos, eos):
        super(Decoder, self).__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        self.dev = dev

        self.sos = sos
        self.eos = eos
        self.pad = pad

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def _loop(self, h_t, inp):
        out, h_next = self.gru(inp , h_t)
        pred = self.out(out)
        return pred, out, h_next

    def teacher_force(self, inputs, h_0):
        inputs.to(self.dev)
        embed = self.embedding(inputs)
        output = []
        h_t = h_0
        for i in range(len(embed[0])):
            pred, out, h_t = self._loop(h_t, embed[:, i])
            output.append(pred.argmax().numpy())
        return output

    def forward(self, h_0, max_n):
        sos = self.embedding(torch.LongTensor(self.sos).to(self.dev))
        output = []
        pred, out, h_t = self._loop(h_0, sos)
        output.append(pred.argmax().numpy())
        n = 1
        while output[-1] != self.eos or n > max_n:
            pred, out, h_t = self._loop(h_t, out)
            output.append(pred.argmax().numpy())
            n += 1
        return output