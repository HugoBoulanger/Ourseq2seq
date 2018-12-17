##
##Générateur de petites histoires

import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import pickle
import re
#from seq2seq import Encoder,Decoder

##Fonctions Traitements des données
def clean_str(string, tolower=True):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if tolower:
        string = string.lower()
    return string.strip()

def voc(data):
    """
    Construction du vocabulaire (et de son vocabulaire inversée)
    data : liste de séquences
    """
    dict1 = {} #vocabulaire
    dict2 = {} #vocabulaire inversé
    dict3 = {} #occurence des mots
    lim = 1 #occurence limite en dessous de laquelle le mot n'est pas pris en compte
    unkw = '<unkwd>' #mot inconnu
    for l in data:
        l2 = clean_str(l) #tokenisation
        for w in l2:
            if w not in dict1.keys():
                dict1[w] = len(dict1)+1
                dict2[len(dict2)+1] = w
                dict3[w] = 1
            else:
                dict3[w] += 1
    #nettoyage dictionnaire
    dict1[unkw] = len(dict1)+1
    dict2[len(dict2)+1] = unkw
    for w,count in dict3.items():
        if count < lim:
            #on remplace le mot dans le dictionnaire par "unknown"
            dict2[dict1[w]] = unkw
            dict1[w] = dict1[unkw]
    return dict1,dict2

def convert(data, vocab):
    """
    Conversion du texte donnée dans le vocabulaire fourni
    data : liste de séquences
    vocab : dictionnaire word -> int ou int -> word
    """
    d = []
    unkw = '<unk>' #mot inconnu
    for l in data:
        c = []
        for w in l:
            if w not in vocab:
                c.append(vocab[unkw])
            else:
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
    
def posttreatement(phrase_c,vocab_inv):
    """
    phrase_th : tensor de la séquence à post-traiter
    vocab_inv : dictionnaire int -> word
    """
    phrase = convert(phrase_c,vocab_inv)
    phrase[0].reverse()
    return " ".join(phrase[0])

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

def callGRU(encoder,decoder,text,vocab,context = None,taille = 5):
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
    ma = 1000 #taille maximal d'une séquence
    text_c = pretreatement(text,vocab,vocab[pad],ma)[0]
    var_taille = 0#random.randint(0,2*taille) - taille
    # print(f"nombre de mots : {taille} + {var_taille}")
    # print(f"text_c : {text_c}")
    # Transformation en Tensor (prise en compte du contexte
    text_f = torch.LongTensor([text_c])
    len_f = torch.LongTensor([len(text_c)])
    # ☻ CALL GRU ☻
    _,exit_enc = encoder.forward(text_f,len_f)
    if(context is not None):
        exit_enc = (exit_enc + context)/2
    exit_dec, ht = decoder.forward(exit_enc,taille+var_taille) #texte de 5 phrases
    exit_dec_f = [[exit_dec[i][j].tolist() for j in range(len(exit_dec[i]))] for i in range(len(exit_dec))]
    # print(f"new context : {ht}")
    return exit_dec_f, ht
        
def Main(vocab,vocab_inv):
    i = 0
    arret = False #peut-être mis à True par le programme (étape 3)
    contexte = None
    #préparation GRU
    enc = Encoder(len(vocab), 100, 100, 2, 'cuda', vocab[pad])
    dec = Decoder(len(vocab), 100, 100, 2, 'cuda', vocab[pad], vocab[sos], vocab[eos])
    enc.to('cuda')
    dec.to('cuda')
    #chargement des poids
    path_enc = "C:/Users/Kardas/Documents/WIA-Project/model/encoder_{e}.pkl"
    path_dec = "C:/Users/Kardas/Documents/WIA-Project/model/decoder_{e}.pkl"
    encoder_state = torch.load(path_enc)
    decoder_state = torch.load(path_dec)
    enc.load_states(encoder_state)
    enc.eval()
    dec.load_states(dict(decoder_state))
    dec.eval()
    #paramétrage de la taille de la prédiction
    taille = int(input("nombre de mots prédis à la fois ? : "))
    while(not arret):
        phrase = takeInput(i)
        exit_c, contexte = callGRU(enc,dec,phrase,vocab,contexte,taille)
        sortie = posttreatement(exit_c,vocab_inv)
        #sortie = "David Albert Huffman est un petit garçon de 10 ans des plus intelligents. Cependant, son monde cours à sa perte lorsque Poupoune décide de s'emparer de l'Europe, alors en pleine crise politique, pour y imposer son monde rose et fleurissant.Avec son ami Lamy, David va devoir lutter contre des adversaires redoutables pour sauver le monde, entrer au MIT et repousser la plus grande menace du Siècle (pour le moment) pour rétablir l'équilibre dans le rapport de Force." #test enchaînement
        printResult(sortie)
        #contexte = exit_c
        i += 1
        
##Make Vocabulary       
corpus = "Huffman contre les Licornes Magiques. Un texte inutile."
pad = '<pad>'
sos = '<sos>'
eos = '<eos>'
#vocab,vocab_inv = voc([corpus])
vocab_inv = {}
vocab = pickle.load(open('C:/Users/Kardas/Documents/WIA-Project/data/vocab.pkl', 'rb'))
for (k,v) in vocab.items():
    vocab_inv[v] = k
# vocab_inv[len(vocab)] = ""
# vocab[pad] = 0
# vocab_inv[len(vocab)] = sos
# vocab[sos] = len(vocab)
# vocab_inv[len(vocab)] = eos
# vocab[eos] = len(vocab)
print(f"vocab : {len(vocab)}\nvocab_inv : {len(vocab_inv)}\n")

##Launch Appli
Main(vocab,vocab_inv)
#test : 
# Drowling in slow motion
# MY LIFE IS POTATOES !
# The first TechnoMage