from seq2seq import Encoder, Decoder
import torch
import torch.nn as nn
import numpy as np
import time
import pickle
from utils import clean_str, voc, convert, padding, pretreatment, posttreatment


unk = '<unk>'
pad = '<pad>'
sos = '<sos>'
eos = '<eos>'
epochs = 50
batch_size = 12

vocab = pickle.load(open('vocab.pkl', 'rb'))

l_tst_src = pickle.load(open('l_tst_src.pkl', 'rb'))
tst_src_p = pickle.load(open('tst_src_p.pkl', 'rb'))
l_tst_tgt = pickle.load(open('l_tst_tgt.pkl', 'rb'))
tst_tgt_p = pickle.load(open('tst_tgt_p.pkl', 'rb'))

l_trn_src = pickle.load(open('l_trn_src.pkl', 'rb'))
trn_src_p = pickle.load(open('trn_src_p.pkl', 'rb'))
l_trn_tgt = pickle.load(open('l_trn_tgt.pkl', 'rb'))
trn_tgt_p = pickle.load(open('trn_tgt_p.pkl', 'rb'))

tst_src_t = torch.LongTensor(tst_src_p)
tst_tgt_t = torch.LongTensor(tst_tgt_p)
trn_src_t = torch.LongTensor(trn_src_p)
trn_tgt_t = torch.LongTensor(trn_tgt_p)


enc = Encoder(len(vocab), 100, 100, 2, 'cuda', vocab[pad])
dec = Decoder(len(vocab), 100, 100, 2, 'cuda', vocab[pad], vocab[sos], vocab[eos], vocab[unk])
enc.to('cuda')
dec.to('cuda')
opt_enc = torch.optim.Adam(enc.parameters())
opt_dec = torch.optim.Adam(dec.parameters())

n_batch = len(trn_src_p)//batch_size

for e in range(epochs):
    enc.train()
    dec.train()
    epoch_loss = 0
    for i in range(n_batch):
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        lengths = torch.LongTensor(l_trn_src[batch_size * i:batch_size * (i + 1)])
        out, h_n = enc(trn_src_t[batch_size * i:batch_size * (i + 1)], lengths)
        output = dec.teacher_force(trn_tgt_t[batch_size * i:batch_size * (i + 1)].reshape([batch_size, tgt_max, 1]),
                                   h_n, torch.LongTensor(l_trn_tgt[batch_size * i:batch_size * (i + 1)]))
        loss = 0
        for o, l, t in zip(output, l_trn_tgt[batch_size * i:batch_size * (i + 1)],
                           trn_tgt_t[batch_size * i:batch_size * (i + 1)]):
            loss += torch.nn.functional.cross_entropy(o[:(l - 1)], t[1:l].to('cuda'))
        loss = loss / batch_size
        enc.zero_grad()
        dec.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        print(f'Batch loss : {loss.item()}')
        nn.utils.clip_grad_norm_(enc.parameters(), 5)
        nn.utils.clip_grad_norm_(dec.parameters(), 5)
        trn_src_t.detach()
        trn_tgt_t.detach()
        opt_dec.step()
        opt_enc.step()

    opt_enc.zero_grad()
    opt_dec.zero_grad()
    torch.save(enc.state_dict(), f'encoder_{e}.pkl')
    torch.save(dec.state_dict(), f'decoder_{e}.pkl')
    print(f'Epoch training loss : {epoch_loss / n_batch}')

    enc.eval()
    dec.eval()
    test_loss = 0
    for i in range(len(tst_tgt_t) // batch_size):
        lengths = torch.LongTensor(l_tst_src[batch_size * i:batch_size * (i + 1)])
        out, h_n = enc(tst_src_t[batch_size * i:batch_size * (i + 1)], lengths)
        output = dec.teacher_force(tst_tgt_t[batch_size * i:batch_size * (i + 1)].reshape([batch_size, tgt_max, 1]),
                                   h_n, torch.LongTensor(l_tst_tgt[batch_size * i:batch_size * (i + 1)]))
        for o, l, t in zip(output, l_tst_tgt[batch_size * i:batch_size * (i + 1)],
                           tst_tgt_t[batch_size * i:batch_size * (i + 1)]):
            test_loss += torch.nn.functional.cross_entropy(o[:(l - 1)], t[1:l].to('cuda')).detach()
        tst_src_t.detach()
        tst_tgt_t.detach()

    print(f'Test loss : {test_loss.item()/len(tst_tgt_t)}')