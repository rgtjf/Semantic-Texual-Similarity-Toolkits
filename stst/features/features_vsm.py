# coding: utf8
from features import Feature
from collections import Counter
import dict_utils
import nltk, math
import lib.kernel.vector_kernel as vk


def vsm(seq1, seq2):
    set_seq1 = list(set(seq1))
    set_seq2 = list(set(seq2))
    bow = set(set_seq1 + set_seq2)
    w2i = {w: i for i, w in enumerate(bow)}

    vec1 = [float(0)] * len(bow)
    for w in set_seq1:
        vec1[w2i[w]] = 1

    vec2 = [float(0)] * len(bow)
    for w in set_seq1:
        vec2[w2i[w]] = 1

    vec1 = vk.normalize(vec1)
    vec2 = vk.normalize(vec2)

    # print(w2i)

    # feats, infos = vk.get_all_kernel(vec1, vec2)
    feats = vk.get_linear_kernel(vec1, vec2)[0] + vk.get_stat_kernel(vec1, vec2)[0]
    infos = [ 'get_linear_kernel', 'get_stat_kernel' ]
    return feats, infos


def vsm_bow(seq1, seq2, topic, convey='idf'):
    idf_weight = dict_utils.DictLoader().load_dict('idf')


    vec1 = [float(0)] * len(topic)
    for w in seq1:
        if convey == 'idf':
            vec1[topic.get(w,0)] += idf_weight.get(w, 10.0)
        else:
            vec1[topic.get(w,0)] += 1

    vec2 = [float(0)] * len(topic)
    for w in seq2:
        if convey == 'idf':
            vec1[topic.get(w,0)] += idf_weight.get(w, 10.0)
        else:
            vec1[topic.get(w,0)] += 1

    vec1 = vk.normalize(vec1)
    vec2 = vk.normalize(vec2)

    # print(w2i)

    # feats, infos = vk.get_all_kernel(vec1, vec2)
    feats = vk.get_linear_kernel(vec1, vec2)[0] + vk.get_stat_kernel(vec1, vec2)[0]
    infos = [ vec1, vec2 ]
    return feats, infos


def make_ngrams(sent, n):
    rez = [sent[i:(-n + i + 1)] for i in range(n - 1)]
    rez.append(sent[n - 1:])
    return zip(*rez)


def ngrams_feats(sa, sb, n, bow=None):
    sa = make_ngrams(sa, n)
    sb = make_ngrams(sb, n)

    features, infos = [], []
    # features.append(jaccrad_coeff(sa, sb))
    # features.append(dice_coeff(sa, sb))
    # features.append(overlap_coeff(sa, sb))
    # features.append(overlap_coeff(sb, sa))
    # features.append(overlap_f1(sb, sa))
    # infos += ['jaccrad_coeff', 'dice_coeff', 'overlap_coeff', 'overlap_coeff', 'overlap_f1']

    if bow:
        feature_vsm, info_vsm = vsm_bow(sa, sb, bow)
    else:
        feature_vsm, info_vsm = vsm(sa, sb)

    features += feature_vsm
    infos += info_vsm

    return features, infos

class TopicFeature(Feature):

    def extract(self, train_instance):
        topic = dict_utils.DictLoader().load_dict('topic')
        sa, sb = train_instance.get_word(type='word')
        sa = [ topic.get(w, 0) for w in sa ]
        sb = [ topic.get(w, 0) for w in sb]

        features, infos = [], [ sa, sb ]
        for n in range(1):
            feature, info = ngrams_feats(sa, sb, n + 1, topic)
            features += feature
        infos += info
        return features, infos

class VSMFeature(Feature):
    def __init__(self, ):
        super

    def extract(self, train_instance):
        lemma_sa, lemma_sb = train_instance.get_word(type='lemma', stopwords=True, lower=True)
        features, infos = [], ['lemma', '1', '2', '3']
        for n in range(3):
            feature, info = ngrams_feats(lemma_sa, lemma_sb, n+1)
            features += feature
        infos += info
        return features, infos
