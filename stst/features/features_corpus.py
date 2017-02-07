# coding: utf8

import numpy
from numpy.linalg import norm
from collections import Counter, defaultdict

import dict_utils
from features import Feature
import config


class Sim:
    def __init__(self, words, vectors):
        words = config.EX_DICT_DIR + '/' + words
        vectors =  config.EX_DICT_DIR + '/' + vectors
        self.word_to_idx = {a: b for b, a in
                            enumerate(w.strip() for w in open(words))}
        self.mat = numpy.loadtxt(vectors)

    def bow_vec(self, b):
        vec = numpy.zeros(self.mat.shape[1])
        for k, v in b.iteritems():
            idx = self.word_to_idx.get(k, -1)
            if idx >= 0:
                vec += self.mat[idx] / (norm(self.mat[idx]) + 1e-8) * v
        return vec

    def calc(self, b1, b2):
        v1 = self.bow_vec(b1)
        v2 = self.bow_vec(b2)
        return abs(v1.dot(v2) / (norm(v1) + 1e-8) / (norm(v2) + 1e-8))

    def calc_kernel(self, b1, b2):
        v1 = self.bow_vec(b1)
        v2 = self.bow_vec(b2)
        import lib.kernel.vector_kernel
        feats, infos = lib.kernel.vector_kernel.get_all_kernel(v1, v2)
        return feats

dict_sim = {}
def get_sim():
    if 'nyt_sim' not in dict_sim:
        nyt_sim = Sim('Word-nyt.txt', 'Vector-nyt.txt')
        dict_sim['nyt_sim'] = nyt_sim
    if 'wiki_sim' not in dict_sim:
        wiki_sim = Sim('Word-wiki.txt', 'Vector-wiki.txt')
        dict_sim['wiki_sim'] = wiki_sim

    nyt_sim = dict_sim['nyt_sim']
    wiki_sim = dict_sim['wiki_sim']
    return nyt_sim, wiki_sim


class DistSim(Feature):
    def extract(self, train_instance):
        nyt_sim, wiki_sim = get_sim()
        idf_weight = dict_utils.DictLoader().load_dict('idf')
        # idf_weight = dict_utils.DictLoader().load_idf_dict()

        def dist_sim(sim, sa, sb):
            wa = Counter(sa)
            wb = Counter(sb)
            d1 = {x: 1 for x in wa}
            d2 = {x: 1 for x in wb}
            return sim.calc_kernel(d1, d2)

        def weighted_dist_sim(sim, sa, sb):
            wa = Counter(sa)
            wb = Counter(sb)
            wa = {x: idf_weight.get(x, 10.0) * wa[x] for x in wa}
            wb = {x: idf_weight.get(x, 10.0) * wb[x] for x in wb}
            return sim.calc_kernel(wa, wb)

        lemma_sa, lemma_sb = train_instance.get_word(type='lemma', lower=True)

        nyt_dist = dist_sim(nyt_sim, lemma_sa, lemma_sb)
        wiki_dist = dist_sim(wiki_sim, lemma_sa, lemma_sb)

        idf_nyt_dist = weighted_dist_sim(nyt_sim, lemma_sa, lemma_sb)
        idf_wiki_dist = weighted_dist_sim(wiki_sim, lemma_sa, lemma_sb)
        # feature = [nyt_dist, wiki_dist, idf_nyt_dist, idf_wiki_dist ]
        feature =  idf_nyt_dist + idf_wiki_dist
        info = [ 'idf-nyt', 'idf-wiki'] # 'nyt', 'wiki',
        return feature, info
