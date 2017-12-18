# coding:utf8
"""
@author rgtjf

==============
@Update 171218
only in STS
import vector_kernel and modify the vector similarity

==============
@Update 171121
1. ADD function get_time_name

==============
@Update 170924
1. ADD function logging part
2. ADD function config part
3. ADD Class DictVocab

==============
@Update 170811
1. ADD      load_embedding_from_text
    - from raw embedding
2. MODIFY   load_word_embedding
    - minor update

==============
@Update 170804
Version 1.0
"""
from __future__ import print_function

import datetime
import io
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
import array
import pyprind

from stst.libs.kernel import vector_kernel as vk

logger = logging.getLogger(__name__)


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


def get_time_name(prefix):
    """
    Returns:
        "prefix_mm_dd_hh_MM"
    """
    time_str = datetime.datetime.now().strftime('_%m%d_%H_%M')
    return prefix + time_str

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
                print("{}\t{}".format(key, vocab[key]), file=fw)
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


def idf_calculator(sentence_list, min_cnt=1):
    """
    idf_calculator
    Args:
        sentence_list: [[w1, w2,...], ...]
        min_cnt: int
    Returns:
        idf_dict: {w:idf}
    """
    doc_num = 0
    word_list = []
    for sequence in sentence_list:
        word_list += sequence
        doc_num += 1
    word_count = Counter()
    for word in word_list:
        word_count[word] += 1
    # filter the word which counts less than min_cnt
    idf_dict = {}
    good_keys = [v for v in word_count.keys() if word_count[v] >= min_cnt]
    # frequence dict
    for key in good_keys:
        idf_dict[key] = word_count[key]
    # idf dict
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


def vector_similarity(vec1, vec2, normlize=True):
    """
    Next is a example:
    Args:   
        vec1 = [0, 1]
        vec2 = [1, 0]
    Returns:
        ['1.414', '1.0', ...], ['euclidean', 'cosine', ...]
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


def sentence_vectorize_features(sa, sb, idf_weight, vocab=None, convey='idf'):
    """
    ensure idf_weight contains all words in seq1 and seq2
    to achieve this, idf_weight format should be the same with seq1
    e.g., train_instance.get_word(type='lemma', lower=True)
    Args: 
        idf_weight: dict
        vocab: dict,    e.g., vocab = word2index(idf_weight)
        convey: 'idf' or 'count'
    :return:
    """
    if not vocab:
        # alert! this op is time-consuming because this will happpen when solving each sentence pair.
        vocab = word2index(idf_weight)
    vec1 = vectorize(sa, idf_weight, vocab, convey)
    vec2 = vectorize(sb, idf_weight, vocab, convey)
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


def create_write_file(file_name, mode='w', encoding='utf8'):
    path = os.path.split(file_name)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    return codecs.open(file_name, mode, encoding=encoding)


def create_read_file(file_name, mode='r', encoding='utf8'):
    return codecs.open(file_name, mode, encoding=encoding)


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

def load_word_embedding(vocab, emb_file, pad_word='__PAD__', unk_word='__UNK__'):
    """
    Pros:
        Save the oov words in oov.p for further analysis.
        Add {0: pad_word, 1: unk_word} into vocab if needed
    Refs:
        class Vectors, https://github.com/pytorch/text/blob/master/torchtext/vocab.py
    Args:
        vocab: dict,
        emb_file: file, path to file of pre-trained word2vec/glove/fasttext
        pad_word:
        unk_word:
    Returns:
        vocab, vectors
    """
    if pad_word not in vocab:
        vocab = {k: v+2 for k, v in vocab.items()}
        vocab[pad_word] = 0
        vocab[unk_word] = 1

    pre_trained = {}
    n_words = len(vocab)
    embeddings = None  # (n_words, n_dim)

    # str call is necessary for Python 2/3 compatibility, since
    # argument must be Python 2 str (Python 3 bytes) or
    # Python 3 str (Python 2 unicode)
    itos, vectors, dim = [], array.array(str('d')), None

    # Try to read the whole file with utf-8 encoding.
    binary_lines = False
    try:
        with io.open(emb_file, encoding="utf8") as f:
            lines = [line for line in f]
    # If there are malformed lines, read in binary mode
    # and manually decode each word from utf-8
    except:
        logger.warning("Could not read {} as UTF8 file, "
                        "reading file as bytes and skipping "
                        "words with malformed UTF8.".format(emb_file))
        with open(emb_file, 'rb') as f:
            lines = [line for line in f]
        binary_lines = True

    logger.info("Loading vectors from {}".format(emb_file))

    process_bar = pyprind.ProgPercent(len(lines))
    for line in lines:
        process_bar.update()
        # Explicitly splitting on " " is important, so we don't
        # get rid of Unicode non-breaking spaces in the vectors.
        entries = line.rstrip().split(b" " if binary_lines else " ")

        word, entries = entries[0], entries[1:]
        if dim is None and len(entries) > 1:
            dim = len(entries)
            # init the embeddings
            embeddings = np.random.uniform(-0.25, 0.25, (n_words, dim))
            embeddings[0,] = np.zeros(dim)

        elif len(entries) == 1:
            logger.warning("Skipping token {} with 1-dimensional "
                            "vector {}; likely a header".format(word, entries))
            continue
        elif dim != len(entries):
            raise RuntimeError(
                "Vector for token {} has {} dimensions, but previously "
                "read vectors have {} dimensions. All vectors must have "
                "the same number of dimensions.".format(word, len(entries), dim))

        if binary_lines:
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')

            except:
                logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                continue

        if word in vocab and word not in pre_trained:
            embeddings[vocab[word]] = [float(x) for x in entries]
            pre_trained[word] = 1

    pre_trained_len = len(pre_trained)
    print('Pre-trained: {}/{} {:.2f}'.format(pre_trained_len, n_words, pre_trained_len * 100.0 / n_words))

    oov_word_list = [w for w in vocab if w not in pre_trained]
    print('oov word list example (30): ', oov_word_list[:30])
    pickle.dump(oov_word_list, open('./oov.p', 'wb'))

    embeddings = np.array(embeddings, dtype=np.float32)
    return vocab, embeddings


def load_embedding_from_text(emb_file, pad_word='__PAD__', unk_word='__UNK__'):
    """
    Add {0: pad_word, 1: unk_word} into stoi
    Note: didn't support GoogleNews-vectors-negative300.bin. 
        This can firstly transformed into GoogleNews-vectors-negative300.txt using `bin2plain.py`.
    Refs:
        class Vectors, https://github.com/pytorch/text/blob/master/torchtext/vocab.py
    Args:
        emb_file: directory for cached vectors
        pad_word:
        unk_word:
    Returns:
        stoi, vectors
    """
    # str call is necessary for Python 2/3 compatibility, since
    # argument must be Python 2 str (Python 3 bytes) or
    # Python 3 str (Python 2 unicode)
    itos, vectors, dim = [], array.array(str('d')), None

    # Try to read the whole file with utf-8 encoding.
    binary_lines = False
    try:
        with io.open(emb_file, encoding="utf8") as f:
            lines = [line for line in f]
    # If there are malformed lines, read in binary mode
    # and manually decode each word from utf-8
    except:
        logger.warning("Could not read {} as UTF8 file, "
                        "reading file as bytes and skipping "
                        "words with malformed UTF8.".format(emb_file))
        with open(emb_file, 'rb') as f:
            lines = [line for line in f]
        binary_lines = True

    logger.info("Loading vectors from {}".format(emb_file))

    process_bar = pyprind.ProgPercent(len(lines))
    for line in lines:
        process_bar.update()
        # Explicitly splitting on " " is important, so we don't
        # get rid of Unicode non-breaking spaces in the vectors.
        entries = line.rstrip().split(b" " if binary_lines else " ")

        word, entries = entries[0], entries[1:]
        if dim is None and len(entries) > 1:
            dim = len(entries)
            # add pad_word
            itos.append(pad_word)
            vectors.extend(np.zeros(dim, ))
            # add unk_word
            itos.append(unk_word)
            vectors.extend(np.random.uniform(-0.25, 0.25, (dim, )))

        elif len(entries) == 1:
            logger.warning("Skipping token {} with 1-dimensional "
                            "vector {}; likely a header".format(word, entries))
            continue
        elif dim != len(entries):
            raise RuntimeError(
                "Vector for token {} has {} dimensions, but previously "
                "read vectors have {} dimensions. All vectors must have "
                "the same number of dimensions.".format(word, len(entries), dim))

        if binary_lines:
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                continue
        vectors.extend(float(x) for x in entries)
        itos.append(word)

    stoi = {word: i for i, word in enumerate(itos)}
    vectors = np.array(vectors, dtype=np.float32).reshape((-1, dim))
    return stoi, vectors


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