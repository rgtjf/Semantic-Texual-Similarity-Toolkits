# coding: utf8
from __future__ import print_function

from collections import Counter

from stst.features.features import Feature
from stst import dict_utils, utils


class nGramOverlapFeature(Feature):

    def __init__(self, type='word', **kwargs):
        super(nGramOverlapFeature, self).__init__(**kwargs)
        self.type = type
        self.feature_name += '-{}'.format(self.type)


    def extract(self, train_instance):
        sa, sb = train_instance.get_pair(self.type)
        overlap_coeff = utils.overlap_coeff(sa, sb)
        unigram_overlap = utils.ngram_match(sa, sb, 1)
        # bigram_overlap  = utils.ngram_match(sa, sb, 2)
        # trigram_overlap = utils.ngram_match(sa, sb, 3)
        # features = [unigram_overlap, bigram_overlap, trigram_overlap]
        features = [overlap_coeff]  # , unigram_ove rlap]
        infos = [sa, sb]
        return features, infos


class nCharGramOverlapFeature(Feature):

    def __init__(self, type='char', **kwargs):
        super(nCharGramOverlapFeature, self).__init__(**kwargs)
        self.type = type
        self.feature_name += '-{}'.format(self.type)


    def extract(self, train_instance):
        sa, sb = train_instance.get_pair('char')
        overlap_coeff = utils.overlap_coeff(sa, sb)
        # unigram_overlap = utils.ngram_match(sa, sb, 1)
        # bigram_overlap  = utils.ngram_match(sa, sb, 2)
        # print(' '.join(sa), ' '.join(sb))
        # trigram_overlap = utils.ngram_match(sa, sb, 3)
        # features = [unigram_overlap, bigram_overlap]
        features = [overlap_coeff]
        infos = [sa, sb]
        return features, infos