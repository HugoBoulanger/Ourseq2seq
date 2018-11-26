import torch.nn as nn
import torch
import torchtext
import spacy

nlp = spacy.load('en_core_web_sm')

train_src = open('writingPrompts/train.wp_source')
train_tgt = open('writingPrompts/train.wp_target')

class Encoder(nn.Module):
    def __init__(self):
        ...

    def forward(self, *input):
        ...

class Decoder(nn.Module):
    def __init__(self):
        ...

    def forward(self, *input):
        ...