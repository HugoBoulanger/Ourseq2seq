##
##Générateur de petites histoires

import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

##Fonctions Traitements des données
def voc(data):
    """
    Construction du vocabulaire (et de son vocabulaire inversée)
    """
    dict1 = {}
    dict2 = {}
    for l in data:
        l2 = l.split(" ")
        for w in l2:
            if w not in dict1.keys():
                dict1[w] = len(dict1)
                dict2[len(dict2)] = w
    return dict1,dict2

def convert(data, vocab):
    """
    Conversion du texte donnée dans le vocabulaire fourni
    data : liste de séquences
    vocab : dictionnaire word -> int
    """
    d = []
    for l in data:
        c = []
        for w in l:
            c.append(vocab[w])
        c.reverse()
        d.append(c)
    return d

def padding(data_c, pad, ma):
    """
    data_c : liste d'int-list
    pad : indice du mot "<pad>" dans le vocabulaire
    ma : taille maximal des séquences paddées
    """
    l = [len(e) for e in data_c]
    m = min(np.amax(l), ma)
    #print(m)
    for i in range(len(l)):
        if l[i] > m:
            l[i] = m
            data_c[i] = data_c[i][:m]
        else:
            for j in range(m-l[i]):
                data_c[i].append(pad)
    z = list(zip(l, data_c))
    z.sort(reverse=True)
    l, data_c = zip(*z)
    return data_c

##Pré & Post Traitement
def pretreatement(phrase,vocab,pad,ma):
    """
    phrase : séquence à pré-traiter
    vocab : dictionnaire int -> word
    pad : indice du mot "<pad>" dans le vocabulaire
    ma : taille maximal des séquences paddées 
    """
    p_c = convert([phrase.split(" ")],vocab)
    sortie = padding(p_c,pad,ma)
    return sortie
    
def posttreatement(phrase_th,vocab_inv):
    """
    phrase_th : tensor de la séquence à post-traiter
    vocab_inv : dictionnaire int -> word
    """
    phrase_c = phrase_th.numpy()
    phrase = convert(phrase_c,vocab_inv)
    return phrase

##Fonctions d'Application
def takeInput(index):
    """
    retourne la phrase entrée par l'utilisateur
    index : indice d'itération de l'application (int >= 0)
    """
    if(index == 0):
        sortie = input("Commencez le récit...\n\n")
    else:
        sortie = input("Poursuivez le récit...\n\n")
    return sortie
    
def printResult(text):
    """
    affiche la séquence de sortie. 1 phrase par ligne.
    """
    sep = text.split(". ")
    for elt in sep:
        if(elt[-1] != "."):
            print(elt+".")
        else:
            print(elt)
    print("\n")

def callGRU(encoder,decoder,text,vocab,context = None,taille = 5):
    """
    INPUT : 
        encoder : modèle de l'encodeur GRU
        decoder : modèle du décodeur GRU
        text : séquence d'entrée du modèle
        vocab : dictionnaire int -> word
        context : séquence précédente du modèle (non pris en compte sir vaut [None])
        taille : nombre de phrases prédites à la fois (nombre de répétition du GRU décodeur sur lui-même)
    OUTPUT : 
        exit_dec : séquences prédites par le modèle GRU en fonction de l'entrée et du contexte
    """
    ma = 1000 #taille maximal d'une séquence
    text_c = pretreatement(text,vocab,vocab[pad],ma)
    # Transformation en Tensor (prise en compte du contexte
    if(context != None):
        text_f = torch.LongTensor(text_c.extend(context))
        len_f = torch.LongTensor(len(text_c)+len(context))
    else:
        text_f = torch.LongTensor(text_c)
        len_f = torch.LongTensor((len(text_c)))
    # ☻ CALL GRU ☻
    exit_enc = encoder.forward(text_f,len_f)
    exit_dec = decoder.forward(exit_enc,taille) #texte de 5 phrases
    print(exit_dec)
    return exit_dec
        
def Main(vocab,vocab_inv):
    i = 0
    arret = False
    contexte = None
    #préparation GRU
    enc = Encoder(len(vocab),20,50,1,'cuda',vocab[pad])
    dec = Decoder(len(vocab), 20, 50, 1, 'cuda', vocab[pad],vocab[sos], vocab[eos])
    enc.to('cuda')
    dec.to('cuda')
    # opt_enc = torch.optim.SparseAdam(enc.parameters())
    # opt_dec = torch.optim.SparseAdam(dec.parameters())
    #paramétrage de la taille de la prédiction
    taille = int(input("nombre de phrases prédites à la fois ? : "))
    while(not arret):
        phrase = takeInput(i)
        exit_c = callGRU(enc,dec,phrase,vocab,contexte,taille)
        sortie = posttreatement(exit_c,vocab_inv)
        print(sortie)
        sortie = "David Albert Huffman est un petit garçon de 10 ans des plus intelligents. Cependant, son monde cours à sa perte lorsque Poupoune décide de s'emparer de l'Europe, alors en pleine crise politique, pour y imposer son monde rose et fleurissant.Avec son ami Lamy, David va devoir lutter contre des adversaires redoutables pour sauver le monde, entrer au MIT et repousser la plus grande menace du Siècle (pour le moment) pour rétablir l'équilibre dans le rapport de Force." #test enchaînement
        printResult(sortie)
        contexte = exit_c
        i += 1
        
##Make Vocabulary       
corpus = "Huffman contre les Licornes Magiques. Un texte inutile."
pad = '<pad>'
sos = '<sos>'
eos = '<eos>'
vocab,vocab_inv = voc([corpus])
vocab[pad] = len(vocab)
vocab[sos] = len(vocab)
vocab[eos] = len(vocab)
print(f"vocab : {vocab}\nvocab_inv : {vocab_inv}\n")

##Launch Appli
Main(vocab,vocab_inv)