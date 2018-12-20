import re

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
    dict1 = {}  # vocabulaire
    dict2 = {}  # vocabulaire inversé
    dict3 = {}  # occurence des mots
    lim = 1  # occurence limite en dessous de laquelle le mot n'est pas pris en compte
    unkw = '<unkwd>'  # mot inconnu
    for l in data:
        l2 = clean_str(l)  # tokenisation
        for w in l2:
            if w not in dict1.keys():
                dict1[w] = len(dict1) + 1
                dict2[len(dict2) + 1] = w
                dict3[w] = 1
            else:
                dict3[w] += 1
    # nettoyage dictionnaire
    dict1[unkw] = len(dict1) + 1
    dict2[len(dict2) + 1] = unkw
    for w, count in dict3.items():
        if count < lim:
            # on remplace le mot dans le dictionnaire par "unknown"
            dict2[dict1[w]] = unkw
            dict1[w] = dict1[unkw]
    return dict1, dict2


def clean_voc(vocab, count, t):
    '''
    Vocabulary cleaning of the words appearing less than t times
    :param vocab: vocabulary given by the voc function
    :param count: dictionary containing the association between a token and it's frequency of appearence
    :param t: the threshold under which the word is removed from the vocabulary
    :return: the dictionary containing the reduced vocabulary
    '''
    for w in count.keys():
        if count[w] < t:
            del vocab[w]
    dic = {}
    for w in vocab.keys():
        dic[w] = len(dic)
    return dic

def convert(data, vocab):
    """
    Conversion du texte donnée dans le vocabulaire fourni
    data : liste de séquences
    vocab : dictionnaire word -> int ou int -> word
    """
    d = []
    unkw = '<unk>'  # mot inconnu
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
    # print(m)
    for i in range(len(l)):
        if l[i] > m:
            l[i] = m
            data_c[i] = data_c[i][:m]
        else:
            for j in range(m - l[i]):
                data_c[i].append(pad)
    z = list(zip(l, data_c))
    z.sort(reverse=True)
    l, data_c = zip(*z)
    return data_c


##Pré & Post Traitement
def pretreatment(phrase, vocab, pad, ma):
    """
    phrase : séquence à pré-traiter
    vocab : dictionnaire int -> word
    pad : indice du mot "<pad>" dans le vocabulaire
    ma : taille maximal des séquences paddées
    """
    p_c = convert([phrase.split(" ")], vocab)
    sortie = padding(p_c, pad, ma)
    return sortie


def posttreatment(phrase_c, vocab_inv):
    """
    phrase_th : tensor de la séquence à post-traiter
    vocab_inv : dictionnaire int -> word
    """
    phrase = convert(phrase_c, vocab_inv)
    phrase[0].reverse()
    return " ".join(phrase[0])