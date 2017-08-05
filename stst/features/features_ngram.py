# coding: utf8
from __future__ import print_function

from collections import Counter

from stst.features.features import Feature
from stst import dict_utils, utils


class nGramOverlapFeature(Feature):

    def extract(self, train_instance):
        sa, sb = train_instance.get_word()
        unigram_overlap = utils.ngram_match(sa, sb, 1)
        bigram_overlap  = utils.ngram_match(sa, sb, 2)
        trigram_overlap = utils.ngram_match(sa, sb, 3)
        features = [unigram_overlap, bigram_overlap, trigram_overlap]
        infos = [sa, sb]
        return features, infos