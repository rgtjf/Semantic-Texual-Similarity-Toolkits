import numpy as np
from collections import Counter
import codecs, math
import theano
import scipy


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_prepare_data(batch, nout):
    batch_sa = [sa for sa, sb, score in batch]
    batch_sb = [sb for sa, sb, score in batch]

    g1x, g1mask = prepare_data(batch_sa)
    g2x, g2mask = prepare_data(batch_sb)

    scores = []
    for sa, sb, score in batch:
        temp = np.zeros(nout)
        score = float(score)
        ceil, fl = int(np.ceil(score)), int(np.floor(score))
        if ceil == fl:
            temp[fl - 1] = 1
        else:
            temp[fl - 1] = ceil - score
            temp[ceil - 1] = score - fl
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype=theano.config.floatX)
    return scores, g1x, g1mask, g2x, g2mask


def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype=theano.config.floatX)
    return x, x_mask

my_emb_rep = {
    'glove100': '/home/junfeng/GloVe/glove.6B.100d.txt',
    'glove300': '/home/junfeng/GloVe/glove.840B.300d.txt',
    'word2vec': '/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin',
    # 'paragram300': '/home/junfeng/paragram-embedding/paragram_300_sl999.txt'
    'paragram300': '/home/junfeng/paragram-embedding/paragram-phrase-XXL.txt'
}


def idf_calculator(sentence_list, min_cnt=1):
    """
    gain idf dict
    Could be used for:
    1). weight the word
    2). generate dict from word to index
    3). used for load word embedding
    :param sentence_list:
    :param min_cnt:
    :return:
    """
    doc_num = 0
    word_list = []
    for sequence in sentence_list:
        word_list += sequence
        doc_num += 1

    word_count = Counter()
    for word in word_list:
        word_count[word] += 1

    idf_dict = {}
    good_keys = [v for v in word_count.keys() if word_count[v] >= min_cnt]

    for key in good_keys:
        idf_dict[key] = word_count[key]

    for key in idf_dict.keys():
        idf_dict[key] = math.log(float(doc_num) / float(idf_dict[key])) / math.log(10)

    return idf_dict


def load_word_embedding(w2i, emb_file, ndim, binary=False):
    pre_trained = {}
    nwords = len(w2i)
    embeddings = np.random.uniform(-0.25, 0.25, (nwords, ndim))

    print('Load word embedding: %s' % emb_file)

    if binary is False:
        with open(emb_file, 'r') as f:
            for line in f:
                sp = line.split()
                assert len(sp) == ndim + 1
                w = sp[0]
                emb = [float(x) for x in sp[1:]]
                if w in w2i and w not in pre_trained:
                    embeddings[w2i[w]] = emb
                    pre_trained[w] = 1

    else:
        with open(emb_file, 'rb') as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                w = word
                emb = np.fromstring(f.read(binary_len), dtype='float32').tolist()
                assert len(emb) == ndim
                if w in w2i and w not in pre_trained:
                    embeddings[w2i[w]] = emb
                    pre_trained[w] = 1
    pre_trained_len = len(pre_trained)
    print(('Pre-trained: %d (%.2f%%)' %
           (pre_trained_len, pre_trained_len * 100.0 / nwords)))
    oov_word_list = [w for w in w2i if w not in pre_trained]
    print('oov word list example (30):', oov_word_list[:30])
    return embeddings


def load_STS(train_file):
    with create_read_file(train_file) as f:
        data = []
        for line in f:
            line = line.strip().split('\t')
            score = float(line[2])
            sa, sb = line[0], line[1]
            sa = sa.lower()
            sb = sb.lower()
            sa = sa.split()
            sb = sb.split()
            data.append((sa, sb, score))
    return data


def load_SICK(train_file):
    with create_read_file(train_file) as f:
        data = []
        f.readline()
        for line in f:
            line = line.strip().split('\t')
            score = float(line[3])
            sa, sb = line[1], line[2]
            sa = sa.lower()
            sb = sb.lower()
            sa = sa.split()
            sb = sb.split()
            data.append((sa, sb, score))
    return data


def create_write_file(file_name, mode='w'):
    import os
    path = os.path.split(file_name)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    return codecs.open(file_name, mode, encoding='utf8')


def create_read_file(file_name, mode='r'):
    return codecs.open(file_name, mode, encoding='utf8')


def eval(gold, prdict):
    pearsonr = scipy.stats.pearsonr(gold, prdict)[0] * 100
    return pearsonr


class Params(object):
    pass
