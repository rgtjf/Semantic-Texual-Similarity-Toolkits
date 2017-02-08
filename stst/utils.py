# coding:utf8
from __future__ import print_function

import time
from functools import wraps
import csv, nltk, math
import codecs
from collections import Counter
import stst.lib.kernel.vector_kernel as vk


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
    with open(to_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contents)


def create_write_file(file_name):
    import os
    path = os.path.split(file_name)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    return codecs.open(file_name, 'w', encoding='utf8')


def create_read_file(file_name):
    return codecs.open(file_name, 'r', encoding='utf8')


@fn_timer
def test():
    pass
    # print(func('2016-01-01 01:10:00'))
    # print(getDistrict('f2c8c4bb99e6377d21de71275afd6cd2'))


def jaccrad_coeff(seq1, seq2):
    filtered_seq1 = list(set(seq1))
    filtered_seq2 = list(set(seq2))
    A_union_B = len(set(filtered_seq1 + filtered_seq2))
    A_inter_B = len([token for token in filtered_seq1 if token in filtered_seq2])
    if A_union_B == 0:
        return 0
    return float(A_inter_B) / float(A_union_B)


def dice_coeff(seq1, seq2):
    filtered_seq1 = list(set(seq1))
    filtered_seq2 = list(set(seq2))
    A_inter_B = len([token for token in filtered_seq1 if token in filtered_seq2])
    if len(filtered_seq1) + len(filtered_seq2) == 0:
        return 0
    return 2 * float(A_inter_B) / float(len(filtered_seq1) + len(filtered_seq2))


def overlap_coeff(seq1, seq2):
    filtered_seq1 = list(set(seq1))
    filtered_seq2 = list(set(seq2))
    A_inter_B = len([token for token in filtered_seq1 if token in filtered_seq2])
    if len(filtered_seq1) == 0:
        return 0
    return float(A_inter_B) / float(len(filtered_seq1))


def overlap_f1(seq1, seq2):
    matches = 0.0
    c1 = Counter(seq1)
    info = []
    for ng in seq2:
        if c1[ng] > 0:
            c1[ng] -= 1
            matches += 1
            info.append(ng)
    p, r, f1 = 0., 0., 1.
    if len(seq1) > 0 and len(seq2) > 0:
        p = matches / len(seq1)
        r = matches / len(seq2)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return f1


# vocab_idx_map: word and index mapping, e.g., egg=1
def Vectorize(vocab_idx_map, idf_dict, list_of_word):
    vec = [float(0)] * len(idf_dict.keys())
    vdist = nltk.FreqDist(list_of_word)
    good_keys = [key for key in vdist.keys() if key in idf_dict.keys()]
    word_cnt = 0
    for key in good_keys:
        word_cnt += vdist[key]
    for key in good_keys:
        tf = float(vdist[key]) / float(word_cnt)
        val = tf * idf_dict[key]
        vec[vocab_idx_map[key]] = val
    return vec


def sequence_match_features(seq1, seq2):
    features, infos = [], []
    features.append(jaccrad_coeff(seq1, seq2))
    features.append(dice_coeff(seq1, seq2))
    features.append(overlap_coeff(seq1, seq2))
    features.append(overlap_coeff(seq1, seq2))
    features.append(overlap_f1(seq1, seq2))
    infos += ['jaccrad_coeff', 'dice_coeff', 'overlap_coeff', 'overlap_coeff', 'overlap_f1']
    return features, infos


# min_cnt = 3, 0.65
# min_cnt = 0, 0.6891
# min_cnt = 1, 0.6891
# min_cnt = 2,
def IDFCalculator(sequence_list, min_cnt=1):
    doc_num = 0
    word_list = []
    for sequence in sequence_list:
        word_list += sequence
        doc_num += 1

    vdist = nltk.FreqDist(word_list)
    idf_dict = {}
    good_keys = [v for v in vdist.keys() if vdist[v] >= min_cnt]

    for key in good_keys:
        idf_dict[key] = vdist[key]

    for key in idf_dict.keys():
        idf_dict[key] = math.log(float(doc_num) / float(idf_dict[key])) / math.log(10)

    return idf_dict


def sequence_vector_features(seq1, seq2, idf_weight=None, convey='idf'):
    """
       ensure idf_weight contains all words in seq1 and seq2
       to achieve this, idf_weight format should be the same with seq1
       e.g., train_instance.get_word(type='lemma', lower=True)
    """

    ''' get vocab index '''
    if idf_weight is None:
        bow = set(seq1 + seq2)
        w2i = {w: i for i, w in enumerate(bow)}
        convey = 'count'
    else:
        w2i = {w: i for i, w in enumerate(idf_weight.keys())}

    ''' get idf min_value '''
    if idf_weight is not None and convey == 'idf':
        min_idf_weight = min(idf_weight.values())

    ''' not very safe, Vectorize is more safe'''
    vec1 = [float(0)] * len(w2i)
    for w in seq1:
        if convey == 'idf':
            vec1[w2i[w]] += idf_weight.get(w, min_idf_weight)
        elif convey == 'count':
            vec1[w2i[w]] += 1
        else:
            raise NotImplementedError

    vec2 = [float(0)] * len(w2i)
    for w in seq2:
        if convey == 'idf':
            vec2[w2i[w]] += idf_weight.get(w, min_idf_weight)
        elif convey == 'count':
            vec2[w2i[w]] += 1
        else:
            raise NotImplementedError

    vec1 = vk.normalize(vec1)
    vec2 = vk.normalize(vec2)

    features, info = vk.get_all_kernel(vec1, vec2)
    # vk.get_linear_kernel(vec1, vec2)[0] + vk.get_stat_kernel(vec1, vec2)[0]
    # features = vk.get_linear_kernel(vec1, vec2)[0] + vk.get_stat_kernel(vec1, vec2)[0]
    infos = ['linear_kernel', 'stat_kernel', 'non-linear_kernal', idf_weight.keys()[:10], idf_weight.values()[:10]]
    return features, infos

def sequence_small_vector_features(seq1, seq2, idf_weight, convey='idf'):
    """
    ensure idf_weight contains all words in seq1 and seq2
    to achieve this, idf_weight format should be the same with seq1
    e.g., train_instance.get_word(type='lemma', lower=True)
    """

    ''' get vocab index '''
    bow = set(seq1 + seq2)
    w2i = {w: i for i, w in enumerate(bow)}

    ''' get min_idf_weight as default idf_weight '''
    min_idf_weight = min(idf_weight.values())

    ''' not very safe, Vectorize is more safe'''
    vec1 = [float(0)] * len(w2i)
    for w in seq1:
        if convey == 'idf':
            vec1[w2i[w]] += idf_weight.get(w, min_idf_weight)
        elif convey == 'count':
            vec1[w2i[w]] += 1
        else:
            raise NotImplementedError

    vec2 = [float(0)] * len(w2i)
    for w in seq2:
        if convey == 'idf':
            vec2[w2i[w]] += idf_weight.get(w, 0.0)
        elif convey == 'count':
            vec2[w2i[w]] += 1
        else:
            raise NotImplementedError

    vec1 = vk.normalize(vec1)
    vec2 = vk.normalize(vec2)

    features, info = vk.get_all_kernel(vec1, vec2)
    infos = ['linear_kernel', 'stat_kernel', 'non-linear_kernal', idf_weight.keys()[:10], idf_weight.values()[:10]]
    return features, infos

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


def sequence_edit_distance_features(sa, sb):
    features, infos = [], []
    features.append(longest_common_prefix(sa, sb))
    features.append(longest_common_suffix(sa, sb))
    features.append(longest_common_substring(sa, sb))
    features.append(longest_common_sequence(sa, sb))
    features.append(levenshtein_disttance(sa, sb))
    infos += ['prefix', 'suffix', 'longest_common_substring', 'longest_common_sequence', 'levenshtein_disttance']
    return features, infos


if __name__ == '__main__':
    sa = ["a", "young", "person", "deep", "in", "thought", "."]
    sb = ["a", "young", "man", "deep", "in", "thought", "."]

    print(longest_common_prefix(sa, sb))

    print(overlap_f1(sa, sb))
