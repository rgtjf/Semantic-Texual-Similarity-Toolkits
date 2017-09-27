# coding:utf8
"""
@author rgtjf

@Update 170924
==============
1. ADD function logging part
2. ADD function config part
3. ADD Class DictVocab

@Update 170811
==============
1. ADD      load_embedding_from_text
    - from raw embedding
2. MODIFY   load_word_embedding
    - minor update

@Update 170804
==============
Version 1.0
"""
from __future__ import print_function

import time
import csv, math
import codecs
import logging
import configparser
from functools import wraps
from collections import Counter
import numpy as np
import os
import pickle

import six


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("total time running %s: %s seconds" % (function.__name__, str(t1 - t0)))
        return result

    return function_timer


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if (cls, args) not in instances:
            instances[(cls, args)] = cls(*args, **kwargs)
        return instances[(cls, args)]

    return _singleton


@fn_timer
def Test():
    pass
    # print(func('2016-01-01 01:10:00'))
    # print(getDistrict('f2c8c4bb99e6377d21de71275afd6cd2'))


@singleton
class SingletonTest(object):
    pass


def get_logger(file_name):
    """ return the default logger """
    # Logger Part
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%m-%d %H:%M")

    # add file handle
    fh = logging.FileHandler(file_name)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # add console handle
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def get_config(config_file):
    config = configparser.ConfigParser(allow_no_value=True,
                interpolation=configparser.ExtendedInterpolation())
    config.read(config_file)
    return config


class DictVocab(object):

    @staticmethod
    def load_from_file(file_path, sep='\t'):
        vocab = {}
        with create_read_file(file_path) as f:
            for idx, line in enumerate(f):
                items = line.strip().split(sep)
                if len(items) == 1:
                    vocab[items[0]] = idx
                elif len(items) == 2:
                    vocab[items[0]] = items[1]
                else:
                    raise NotImplementedError
        print('load from FILE {}'.format(file_path))
        return vocab

    @staticmethod    
    def dump_to_file(vocab, file_path, sep='\t', sort_by_key=True, reverse=False):
        with create_write_file(file_path) as fw:
            items = vocab.items()
            if sort_by_key:
                keys = sorted(items, cmp=lambda x: x[0], reverse=reverse)
            else:
                keys = sorted(items, cmp=lambda x: x[1], reverse=reverse)
            for key in keys:
                print("{}\t{}".format(key, vocab[key]), flle=fw)
        print('dump to FILE {}'.format(file_path))


#############################################
# ` Part
# idf_calculator: gain idf from sentence list
#############################################

def split_abbreviation(word):
    res = []
    char = ''
    for ch in word:
        if char != '' and char[-1].islower() and ch.isupper():
            res.append(char)
            char = ''
        char += ch
    if char != '':
        res.append(char)
    return res


def word2char(word_list):
    """
    Translate word_list to char_list
    """
    if type(word_list) is six.text_type:
        word_list = word_list.split()

    char_list = []
    word_string = ''.join(word_list)
    char = ''
    for ch in word_string:
        if ord(ch) < 128:
            char += ch
        else:
            if char != '':
                char_list += split_abbreviation(char)
                char = ''
            char_list.append(ch)
    if char != '': char_list += split_abbreviation(char)
    return char_list


def word2index(word_list):
    """
    return the vocab {w:i} of word_list
    """
    if type(word_list) is list:
        vocab = {word:i for i, word in enumerate(word_list)}
    elif type(word_list) is dict:
        vocab = {word:i for i, word in enumerate(word_list.keys())}
    else:
        raise NotImplementedError
    return vocab


def pos2tag(pos):
    if pos   in ['NN', 'NNS', 'NNP', 'NNPS']:
        pos = 'n'
    elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        pos = 'v'
    elif pos in ['JJ', 'JJR', 'JJS']:
        pos = 'a'
    elif pos in ['RB', 'RBR', 'RBS']:
        pos = 'r'
    else:
        pos = '#'
    return pos


def idf_calculator(sentence_list, min_cnt=1, max_cnt=None):
    doc_num = 0
    word_list = []
    for sequence in sentence_list:
        word_list += sequence
        doc_num += 1

    word_count = Counter()
    for word in word_list:
        word_count[word] += 1

    if max_cnt is None:
        good_keys = [v for v in word_count.keys() if word_count[v] >= min_cnt]
    else:
        good_keys = [v for v in word_count.keys() if word_count[v] >= min_cnt and word_count[v] <= max_cnt]

    idf_dict = {}
    for key in good_keys:
        idf_dict[key] = word_count[key]

    for key in idf_dict.keys():
        idf_dict[key] = math.log(float(doc_num) / float(idf_dict[key])) / math.log(10)

    return idf_dict


def vectorize(sentence, idf_weight, vocab, convey='idf'):
    """
    idf_weight: {word: weight}
    vocab: {word: index}
    """
    vec = np.zeros(len(vocab), dtype=np.float32)
    for word in sentence:
        if word not in vocab:
            continue
        if convey == 'idf':
            vec[vocab[word]] += idf_weight[word]
        elif convey == 'count':
            vec[vocab[word]] += 1
        else:
            raise NotImplementedError
    return vec


def vector_similarity(vec1, vec2):
    """
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
    features = []
    features.append(cosine_distance(vec1, vec2))
    features.append(manhattan_distance(vec1, vec2, norm=True))
    features.append(euclidean_distance(vec1, vec2, norm=True))
    features.append(chebyshev_distance(vec1, vec2, norm=True))
    infos = ['cosine, manhattan', 'euclidean', 'chbyshev']
    return features, infos


#############################################
# ` Part
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
    vocab = word2index(idf_weight)
    vec1 = vectorize(sa, idf_weight, vocab)
    vec2 = vectorize(sb, idf_weight, vocab)
    features, info = vector_similarity(vec1, vec2)
    return features, info


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


def write_dict_to_csv(contents_dict, to_file):
    fieldnames = []
    contents = []
    with open(to_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contents)


def create_write_file(file_name, mode='w'):
    path = os.path.split(file_name)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    return codecs.open(file_name, mode, encoding='utf8')


def create_read_file(file_name, mode='r'):
    return codecs.open(file_name, mode, encoding='utf8')


def check_file_exist(file_path):
    path = os.path.split(file_path)[0]
    if not os.path.exists(path):
        print('===> create path: {}'.format(path))
        os.makedirs(path)


def check_dir_exist(dir_path):
    path = dir_path
    if not os.path.exists(path):
        print('===> create path: {}'.format(path))
        os.makedirs(path)


#############################################
# ` Part IV
# Word Embedding Utils
#############################################


def load_word_embedding(vocab, emb_file, n_dim,
                        pad_word='__PAD__', unk_word='__UNK__'):
    """
    UPDATE_1: fix the word embedding
    ===
    UPDATE_0: save the oov words in oov.p (pickle)
    Pros: to analysis why the this happen !!!
    ===
    :param vocab: dict, vocab['__UNK__'] = 0
    :param emb_file: str, file_path
    :param n_dim:
    :param pad_word
    :param unk_word
    :return: np.array(n_words, n_dim)
    """
    print('Load word embedding: %s' % emb_file)
    assert vocab[pad_word] == 0
    assert vocab[unk_word] == 1

    pre_trained = {}
    n_words = len(vocab)

    embeddings = np.random.uniform(-0.25, 0.25, (n_words, n_dim))
    # embeddings[0, ] = np.zeros(n_dim)

    with codecs.open(emb_file, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f):
            if idx == 0 and len(line.split()) == 2:
                continue
            sp = line.rstrip().split()
            if len(sp) != n_dim + 1:
                print(sp[0:len(sp) - n_dim])

            w = ''.join(sp[0:len(sp) - n_dim])
            emb = [float(x) for x in sp[len(sp) - n_dim:]]

            if w in vocab and w not in pre_trained:
                embeddings[vocab[w]] = emb
                pre_trained[w] = 1

    pre_trained_len = len(pre_trained)

    print('Pre-trained: {}/{} {:.2f}'.format(pre_trained_len, n_words, pre_trained_len * 100.0 / n_words))

    oov_word_list = [w for w in vocab if w not in pre_trained]
    print('oov word list example (30): ', oov_word_list[:30])

    pickle.dump(oov_word_list, open('./oov.p', 'wb'))

    embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings


def load_embedding_from_text(emb_file, n_dim,
                             pad_word='__PAD__', unk_word='__UNK__'):
    """
    :return: embed: numpy, vocab2id: dict
    """
    print('==> loading embed from txt')

    vocab2id = {}
    embed = []
    word_id = 0

    vocab2id[pad_word] = word_id
    embed.append(np.zeros(shape=[n_dim, ], dtype=np.float32))

    word_id += 1
    vocab2id[unk_word] = word_id
    embed.append(np.random.uniform(-0.25, 0.25, size=[n_dim, ]))

    with codecs.open(emb_file, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f):
            if idx == 0 and len(line.split()) == 2:
                print('embedding info: ', line)
                continue
            sp = line.rstrip().split()

            if len(sp) != n_dim + 1:
                print(sp[0:len(sp) - n_dim])

            w = ''.join(sp[0:len(sp) - n_dim])
            emb = [float(x) for x in sp[len(sp) - n_dim:]]

            word_id += 1
            vocab2id[w] = word_id
            embed.append(emb)

    print('==> finished load input embed from txt')
    return np.array(embed, dtype=np.float32), vocab2id


#############################################
# ` Part III
# Vector Operation
#############################################

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def check_pairwise_vector(v1, v2, norm):
    if isinstance(v1, list):
        v1 = np.array(v1)
    if isinstance(v2, list):
        v2 = np.array(v2)
    if v1.shape != v2.shape:
        raise ValueError("v1 and v2 should be of same shape. They were "
                         "respectively %r and %r long." % (v1.shape, v2.shape))
    if norm:
        v1 = normalize(v1)
        v2 = normalize(v2)
    return v1, v2


def cosine_distance(v1, v2, norm=True):
    """
    return cosine distance (NOT similarity)
    """
    v1, v2 = check_pairwise_vector(v1, v2, norm)
    cosine = (v1 * v2).sum()
    if np.isnan(cosine):
        cosine = 1.
    return 1. - cosine


def manhattan_distance(v1, v2, norm=False):
    """
    return ||v1 - v2||_1
    """
    v1, v2 = check_pairwise_vector(v1, v2, norm)
    diff = v1 - v2
    K = np.abs(diff).sum()
    return K


def euclidean_distance(v1, v2, norm=False):
    """
    return ||v1 - v2||_2
    """
    v1, v2 = check_pairwise_vector(v1, v2, norm)
    diff = v1 - v2
    K = np.sqrt((diff ** 2).sum())
    return K


def chebyshev_distance(v1, v2, norm=False):
    """
    return ||v1 - v2||_oo
    """
    v1, v2 = check_pairwise_vector(v1, v2, norm)
    diff = v1 - v2
    K = np.abs(diff).max()
    return K


#############################################
# ` Part II
# Sequence Operation
#############################################

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


#############################################
# ` Part I
# Set Operation
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


def make_ngram(sent, n):
    rez = [sent[i:(-n + i + 1)] for i in range(n - 1)]
    rez.append(sent[n - 1:])
    return list(zip(*rez))


def ngram_match(sa, sb, n):
    nga = make_ngram(sa, n)
    ngb = make_ngram(sb, n)
    f1 = overlap_f1(nga, ngb)
    return f1