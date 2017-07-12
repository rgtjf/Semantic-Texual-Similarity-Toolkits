# coding:utf8
from __future__ import print_function

import time
import csv, math
import codecs
from functools import wraps
from collections import Counter
import numpy as np
import os

from stst.lib.kernel import vector_kernel as vk


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" % (function.__name__, str(t1 - t0)))
        return result

    return function_timer


def singleton(cls):
    instances = {}

    def _singleton(*args, **kw):
        if (cls, args) not in instances:
            instances[(cls, args)] = cls(*args, **kw)
        return instances[(cls, args)]

    return _singleton


def match_f1(matches, length_sa, length_sb):
    p, r, f1 = 0.0, 0.0, 1.0
    if length_sa > 0 and length_sb > 0:
        p = matches / length_sa
        r = matches / length_sb
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return f1


def get_tag(pos):
    if pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS':
        pos = 'n'
    elif pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == "VBP" or pos == "VBZ":
        pos = 'v'
    elif pos == 'JJ' or pos == 'JJR' or pos == 'JJS':
        pos = 'a'
    elif pos == 'RB' or pos == 'RBR' or pos == 'RBS':
        pos = 'r'
    else:
        pos = '#'
    return pos


def write_dict_to_csv(contents_dict, to_file):
    fieldnames = []
    contents = []
    with open(to_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contents)


def create_write_file(file_name, mode='w'):
    import os
    path = os.path.split(file_name)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    return codecs.open(file_name, mode, encoding='utf8')


def create_read_file(file_name, mode='r'):
    return codecs.open(file_name, mode, encoding='utf8')


@fn_timer
def test():
    pass
    # print(func('2016-01-01 01:10:00'))
    # print(getDistrict('f2c8c4bb99e6377d21de71275afd6cd2'))

#############################################
# ` Part I
# Caclulate common parts between sa and sb
#############################################

def jaccrad_coeff(sa, sb):
    filtered_seq1 = list(set(sa))
    filtered_seq2 = list(set(sb))
    A_union_B = len(set(filtered_seq1 + filtered_seq2))
    A_inter_B = len([token for token in filtered_seq1 if token in filtered_seq2])
    if A_union_B == 0:
        return 0
    return float(A_inter_B) / float(A_union_B)


def dice_coeff(sa, sb):
    filtered_seq1 = list(set(sa))
    filtered_seq2 = list(set(sb))
    A_inter_B = len([token for token in filtered_seq1 if token in filtered_seq2])
    if len(filtered_seq1) + len(filtered_seq2) == 0:
        return 0
    return 2 * float(A_inter_B) / float(len(filtered_seq1) + len(filtered_seq2))


def overlap_coeff(sa, sb):
    filtered_seq1 = list(set(sa))
    filtered_seq2 = list(set(sb))
    A_inter_B = len([token for token in filtered_seq1 if token in filtered_seq2])
    if len(filtered_seq1) == 0:
        return 0
    return float(A_inter_B) / float(len(filtered_seq1))


def overlap_f1(sa, sb):
    matches = 0.0
    c1 = Counter(sa)
    info = []
    for ng in sb:
        if c1[ng] > 0:
            c1[ng] -= 1
            matches += 1
            info.append(ng)
    p, r, f1 = 0., 0., 1.
    if len(sa) > 0 and len(sb) > 0:
        p = matches / len(sa)
        r = matches / len(sb)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return f1


def longest_common_suffix(sa, sb):
    l = min(len(sa), len(sb))
    r = l
    for i in range(l):
        idx = l - 1 - i
        if sa[idx] != sb[idx]:
            r = i
            break
    rr = 0.0 if l == 0 else r / l
    return rr


def longest_common_prefix(sa, sb):
    l = min(len(sa), len(sb))
    r = l
    for i in range(l):
        idx = i
        if sa[idx] != sb[idx]:
            r = i
            break
    rr = 0.0 if l == 0 else 1.0 * r / l
    return rr


def longest_common_substring(sa, sb):
    l = min(len(sa), len(sb))
    r = 0
    for i in range(len(sa)):
        for j in range(len(sb)):
            k = 0
            while (i + k < len(sa) and j + k < len(sb)):
                if sa[i + k] != sb[j + k]:
                    break
                k = k + 1
            r = max(r, k)
    rr = 0.0 if l == 0 else 1.0 * r / l
    return rr


def longest_common_sequence(sa, sb):
    la = len(sa)
    lb = len(sb)
    l = min(la, lb)
    dp = [[0] * (lb + 1)] * (la + 1)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if sa[i - 1] == sb[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
    r = dp[la][lb]
    rr = 0.0 if l == 0 else 1.0 * r / l
    return rr


def levenshtein_disttance(sa, sb):
    la = len(sa)
    lb = len(sb)
    l = min(la, lb)
    dp = [[0] * (lb + 1)] * (la + 1)
    for i in range(0, la + 1):
        dp[i][0] = i
    for j in range(0, lb + 1):
        dp[0][j] = j

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if sa[i - 1] == sb[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    r = dp[la][lb]
    rr = 0.0 if l == 0 else 1.0 * r / l
    return rr


def make_ngram(sent, n):
    rez = [sent[i:(-n + i + 1)] for i in range(n - 1)]
    rez.append(sent[n - 1:])
    return list(zip(*rez))


def ngram_match(sa, sb, n):
    nga = make_ngram(sa, n)
    ngb = make_ngram(sb, n)
    f1 = overlap_f1(nga, ngb)
    return f1


#############################################
# ` Part II
# idf_calculator: gain idf from sentence list
#
#############################################

def idf_calculator(sentence_list, min_cnt=1):
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


def vectorize(sentence, idf_weight, word2index, convey='idf'):
    vec = [float(0)] * len(word2index)
    for word in sentence:
        if word not in word2index:
            continue
        if convey == 'idf':
            vec[word2index[word]] += idf_weight[word]
        elif convey == 'count':
            vec[word2index[word]]  += 1
        else:
            raise NotImplementedError
    return vec


def vector_similarity(vec1, vec2, normlize=True):
    """
    :return:
     example:
        vec1 = [0, 1]
        vec2 = [1, 0]
        return: ['1.414', '1.0', ...], ['euclidean', 'cosine', ...]

        which means:
            euclidean 1.41421356237
            cosine 1.0
            manhattan 2
            chebyshev_distance 1
            spearmanr -1.0
            kendalltau -1.0
            pearsonr -1.0
            polynomial 1.0
            rbf 0.493068691395
            laplacian 0.367879441171
            sigmoid 0.761594155956
    """
    if normlize:
        vec1 = vk.normalize(vec1)
        vec2 = vk.normalize(vec2)
    return vk.get_all_kernel(vec1, vec2)


#############################################
# ` Part III
#  merge all above basic functions
#
#############################################

def sentence_match_features(seq1, seq2):
    features, infos = [], []
    features.append(jaccrad_coeff(seq1, seq2))
    features.append(dice_coeff(seq1, seq2))
    features.append(overlap_coeff(seq1, seq2))
    features.append(overlap_coeff(seq1, seq2))
    features.append(overlap_f1(seq1, seq2))
    infos += ['jaccrad_coeff', 'dice_coeff', 'overlap_coeff', 'overlap_coeff', 'overlap_f1']
    return features, infos


def sentence_sequence_features(sa, sb):
    features, infos = [], []
    features.append(longest_common_prefix(sa, sb))
    features.append(longest_common_suffix(sa, sb))
    features.append(longest_common_substring(sa, sb))
    features.append(longest_common_sequence(sa, sb))
    features.append(levenshtein_disttance(sa, sb))
    infos += ['prefix', 'suffix', 'longest_common_substring', 'longest_common_sequence', 'levenshtein_disttance']
    return features, infos


def sentence_vectorize_features(sa, sb, idf_weight, convey='idf'):
    """
       ensure idf_weight contains all words in seq1 and seq2
       to achieve this, idf_weight format should be the same with seq1
       e.g., train_instance.get_word(type='lemma', lower=True)
    :param idf_weight: dict
    :param convey: 'idf' or 'count'
    :return:
    """
    """
       ensure idf_weight contains all words in seq1 and seq2
       to achieve this, idf_weight format should be the same with seq1
       e.g., train_instance.get_word(type='lemma', lower=True)
    """
    word2index = {word: i for i, word in enumerate(idf_weight.keys())}
    vec1 = vectorize(sa, idf_weight, word2index)
    vec2 = vectorize(sb, idf_weight, word2index)
    features, info = vector_similarity(vec1, vec2)

    info = ['linear_kernel', 'stat_kernel', 'non-linear_kernal',
            list(idf_weight.keys())[:10], list(idf_weight.values())[:10]]
    return features, info


#############################################
# ` Part IV
#  word embedding
#
#############################################

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

#
# my_emb_rep = {
#     'glove100': '/home/junfeng/GloVe/glove.6B.100d.txt',
#     'glove300': '/home/junfeng/GloVe/glove.840B.300d.txt',
#     'word2vec': '/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin',
#     'paragram300': '/home/junfeng/paragram-embedding/paragram_300_sl999.txt'
# }
#
#
# def load_w2v_offline(vw):
#     """vw is dict about Vocab of Words
#     """
#     emb_file = my_emb_rep['word2vec']
#     ndim = 300
#     pre_trained = {}
#     nwords = len(vw)
#     embeddings = np.random.uniform(-0.25, 0.25, (nwords, ndim))
#     f = open(emb_file, 'rb')
#     header = f.readline()
#     vocab_size, layer1_size = map(int, header.split())
#     binary_len = np.dtype('float32').itemsize * layer1_size
#     for line in range(vocab_size):
#         word = []
#         while True:
#             ch = f.read(1)
#             if ch == ' ':
#                 word = ''.join(word)
#                 break
#             if ch != '\n':
#                 word.append(ch)
#         w = word
#         emb = np.fromstring(f.read(binary_len), dtype='float32').tolist()
#         assert len(emb) == ndim
#
#         if w in vw and w not in pre_trained:
#             embeddings[vw[w]] = emb
#             pre_trained[w] = 1
#     pre_trained_len = len(pre_trained)
#     print(('Pre-trained: %d (%.2f%%)' %
#            (pre_trained_len, pre_trained_len * 100.0 / nwords)))
#
#     ''' write oov words '''
#     with open('oov.txt', 'w') as f:
#         for w in vw:
#             if w not in pre_trained:
#                 f.write(w + '\n')
#
#     return embeddings
#
#
# def load_embedding_offline(vw):
#     """
#     glove
#     paragram
#     """
#     pre_trained = {}
#     ndim = 300
#     nwords = len(vw)
#     embeddings = np.random.uniform(-0.25, 0.25, (nwords, ndim))
#     textfile = my_emb_rep['paragram300']
#     f = open(textfile, 'r')
#     for line in open(textfile):
#         sp = line.split()
#         assert len(sp) == ndim + 1
#         w = sp[0]
#         emb = [float(x) for x in sp[1:]]
#         if w in vw and w not in pre_trained:
#             embeddings[vw[w]] = emb
#             pre_trained[w] = 1
#         pre_trained_len = len(pre_trained)
#     print(('Pre-trained: %d (%.2f%%)' %
#            (pre_trained_len, pre_trained_len * 100.0 / nwords)))
#
#     ''' write oov words '''
#     with open('oov.txt', 'w') as f:
#         for w in vw:
#             if w not in pre_trained:
#                 f.write(w + '\n')
#     return embeddings



class FileManager(object):

    @classmethod
    def get_file(cls, path):
        path, file = os.path.split(path)
        return file

    @classmethod
    def get_filename(cls, path):
        path, file = os.path.split(path)
        filename = os.path.splitext(file)[0]
        return filename

if __name__ == '__main__':
    sa = ["a", "young", "person", "deep", "in", "thought", "."]
    sb = ["a", "young", "man", "deep", "in", "thought", "."]

    print(longest_common_prefix(sa, sb))

    print(overlap_f1(sa, sb))
