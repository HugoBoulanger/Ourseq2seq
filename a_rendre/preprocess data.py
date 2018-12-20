from utils import clean_str, voc, convert, padding, pretreatment, posttreatment, clean_voc
import pickle

train_source = 'data/train.wp_source'
train_target = 'data/train.wp_target'
test_source = 'data/test.wp_source'
test_target = 'data/test.wp_target'

unk = '<unk>'
pad = '<pad>'
sos = '<sos>'
eos = '<eos>'

# Max size of the target
tgt_max = 50

def read_file(path, max_lines=0):
    '''

    :param path: path of the file
    :param max_lines: max lines to read, default 0 for the whole file
    :return: the list of the lines
    '''
    f = open(path, 'r')

    if max_lines == 0:
        ret = f.readlines()

    else:
        ret = []

        i = 0
        for l in f:
            ret.append(l)
            i += 1
            if i >= max_lines:
                break
    f.close()

    return ret



if __name__=='__main__':
    trn_src = read_file(train_source, max_lines=25000)
    trn_tgt = read_file(train_target, max_lines=25000)
    tst_src = read_file(test_source)
    tst_tgt = read_file(test_target)

    #The offset of 3 is used because the dataset comes with some tags at the beggining of each lines
    for i in range(len(tst_src)):
        tst_src[i] = clean_str(tst_src[i]).split()[3:]
        tst_tgt[i] = clean_str(tst_tgt[i]).split()[3:tgt_max + 3]

    for i in range(len(trn_src)):
        trn_src[i] = clean_str(trn_src[i]).split()[3:]
        trn_tgt[i] = clean_str(trn_tgt[i]).split()[3:tgt_max + 3]

    #Vocabulary preprocessing
    vocab, count = voc(trn_src + trn_tgt)
    vocab = clean_voc(vocab, count, 2)

    vocab[unk] = len(vocab)
    vocab[pad] = len(vocab)
    vocab[sos] = len(vocab)
    vocab[eos] = len(vocab)

    pickle.dump(vocab, open('data/vocab.pkl', 'wb'))

    #Testing set preprocessing
    tst_src_c = convert(tst_src, vocab, unk)
    tst_tgt_c = convert(tst_tgt, vocab, unk)

    tst_tgt_c[:].insert(0, vocab[sos])
    tst_tgt_c[:].append(vocab[eos])

    l_tst_src, tst_src_p = padding(tst_src_c, vocab[eos], vocab[pad], tgt_max)
    l_tst_tgt, tst_tgt_p = padding(tst_tgt_c, vocab[eos], vocab[pad], tgt_max)

    z = list(zip(l_tst_src, tst_src_p, l_tst_tgt, tst_tgt_p))
    z.sort(reverse=True)
    l_tst_src, tst_src_p, l_tst_tgt, tst_tgt_p = zip(*z)

    pickle.dump(l_tst_src, open('data/l_tst_src.pkl', 'wb'))
    pickle.dump(tst_src_p, open('data/tst_src_p.pkl', 'wb'))
    pickle.dump(l_tst_tgt, open('data/l_tst_tgt.pkl', 'wb'))
    pickle.dump(tst_tgt_p, open('data/tst_tgt_p.pkl', 'wb'))

    #Training set preprocessing
    trn_src_c = convert(trn_src, vocab, unk)
    trn_tgt_c = convert(trn_tgt, vocab, unk)

    trn_tgt_c[:].insert(0, vocab[sos])
    trn_tgt_c[:].append(vocab[eos])

    l_trn_src, trn_src_p = padding(trn_src_c, vocab[eos], vocab[pad], tgt_max)
    l_trn_tgt, trn_tgt_p = padding(trn_tgt_c, vocab[eos], vocab[pad], tgt_max)

    z = list(zip(l_trn_src, trn_src_p, l_trn_tgt, trn_tgt_p))
    z.sort(reverse=True)
    l_trn_src, trn_src_p, l_trn_tgt, trn_tgt_p = zip(*z)

    pickle.dump(l_trn_src, open('data/l_trn_src.pkl', 'wb'))
    pickle.dump(trn_src_p, open('data/trn_src_p.pkl', 'wb'))
    pickle.dump(l_trn_tgt, open('data/l_trn_tgt.pkl', 'wb'))
    pickle.dump(trn_tgt_p, open('data/trn_tgt_p.pkl', 'wb'))

