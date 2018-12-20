##
##Générateur de petites histoires

import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import pickle
import re
from seq2seq import Encoder,Decoder
from utils import clean_str, voc, convert, padding, pretreatment, posttreatment


##Fonctions d'Application
def takeInput(index):
    """
    retourne la phrase entrée par l'utilisateur
    index : indice d'itération de l'application (int >= 0)
    """
    if (index == 0):
        sortie = input("Commencez le récit...\n\n")
    else:
        sortie = input("Poursuivez le récit...\n\n")
    return sortie


def printResult(text):
    """
    affiche la séquence de sortie. 1 phrase par ligne.
    """
    # sep = text.split(". ")
    # if(len(sep) > 0):
    #     for elt in sep:
    #         if(elt[-1] != "."):
    #             print(elt+".")
    #         else:
    #             print(elt)
    # else:
    print(text)
    print("\n")


def callGRU(encoder, decoder, text, vocab, context=None, taille=5):
    """
    INPUT :
        encoder : modèle de l'encodeur GRU
        decoder : modèle du décodeur GRU
        text : séquence d'entrée du modèle
        vocab : dictionnaire int -> word
        context : séquence précédente du modèle (non pris en compte sir vaut [None])
        taille : nombre de mots prédis à la fois (nombre de répétition du GRU décodeur sur lui-même)
    OUTPUT :
        exit_dec : séquences prédites par le modèle GRU en fonction de l'entrée et du contexte
    """
    ma = 1000  # taille maximal d'une séquence
    text_c = pretreatment(text, vocab, vocab[pad], ma)[0]
    var_taille = 0  # random.randint(0,2*taille) - taille
    # print(f"nombre de mots : {taille} + {var_taille}")
    # print(f"text_c : {text_c}")
    # Transformation en Tensor (prise en compte du contexte
    text_f = torch.LongTensor([text_c])
    len_f = torch.LongTensor([len(text_c)])
    # ☻ CALL GRU ☻
    _, exit_enc = encoder.forward(text_f, len_f)
    if (context is not None):
        exit_enc = (exit_enc + context) / 2
    exit_dec, ht = decoder.forward(exit_enc, taille + var_taille)  # texte de 5 phrases
    exit_dec_f = [[exit_dec[i][j].tolist() for j in range(len(exit_dec[i]))] for i in range(len(exit_dec))]
    # print(f"new context : {ht}")
    return exit_dec_f, ht


def Main(vocab, vocab_inv):
    i = 0
    arret = False  # peut-être mis à True par le programme (étape 3)
    contexte = None
    # préparation GRU
    enc = Encoder(len(vocab), 100, 100, 2, 'cuda', vocab[pad])
    dec = Decoder(len(vocab), 100, 100, 2, 'cuda', vocab[pad], vocab[sos], vocab[eos], vocab[unk])
    enc.to('cuda')
    dec.to('cuda')
    # chargement des poids
    path_enc = "encoder_9.pkl"
    path_dec = "decoder_9.pkl"
    encoder_state = torch.load(path_enc)
    decoder_state = torch.load(path_dec)
    enc.load_states(encoder_state)
    enc.eval()
    dec.load_states(dict(decoder_state))
    dec.eval()
    # paramétrage de la taille de la prédiction
    taille = int(input("nombre de mots prédis à la fois ? : "))
    while (not arret):
        phrase = takeInput(i)
        exit_c, contexte = callGRU(enc, dec, phrase, vocab, contexte, taille)
        sortie = posttreatment(exit_c, vocab_inv)
        # sortie = "David Albert Huffman est un petit garçon de 10 ans des plus intelligents. Cependant, son monde cours à sa perte lorsque Poupoune décide de s'emparer de l'Europe, alors en pleine crise politique, pour y imposer son monde rose et fleurissant.Avec son ami Lamy, David va devoir lutter contre des adversaires redoutables pour sauver le monde, entrer au MIT et repousser la plus grande menace du Siècle (pour le moment) pour rétablir l'équilibre dans le rapport de Force." #test enchaînement
        printResult(sortie)
        # contexte = exit_c
        i += 1


##Make Vocabulary
corpus = "Huffman contre les Licornes Magiques. Un texte inutile."
pad = '<pad>'
sos = '<sos>'
eos = '<eos>'
unk = '<unk>'
# vocab,vocab_inv = voc([corpus])
vocab_inv = {}
vocab = pickle.load(open('vocab.pkl', 'rb'))
for (k, v) in vocab.items():
    vocab_inv[v] = k
# vocab_inv[len(vocab)] = ""
# vocab[pad] = 0
# vocab_inv[len(vocab)] = sos
# vocab[sos] = len(vocab)
# vocab_inv[len(vocab)] = eos
# vocab[eos] = len(vocab)
print(f"vocab : {len(vocab)}\nvocab_inv : {len(vocab_inv)}\n")

##Launch Appli
Main(vocab, vocab_inv)
# test :
# Drowling in slow motion
# MY LIFE IS POTATOES !
# The first TechnoMage