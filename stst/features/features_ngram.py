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


class WeightednGramOverlapFeature(Feature):

    def __init__(self, type='word', **kwargs):
        super(WeightednGramOverlapFeature, self).__init__(**kwargs)
        self.type = type
        self.feature_name += '-{}'.format(self.type)

    def extract(self, train_instance):
        pos_sa, pos_sb = train_instance.get_list('pos')
        word_sa, word_sb =train_instance.get_list('word')


        weighted_inter = 0
        weighted_seq1 = 0

        for pos, word in zip(pos_sa, word_sa):
            if pos[0] == 'N':
                score = 3.
            else:
                score = 1.
            if word in word_sb:
                weighted_inter += score
            weighted_seq1 += score

        weighted_overlap = weighted_inter / weighted_seq1 if weighted_seq1 != 0 else 0.0

        features = [weighted_overlap]
        infos = [word_sa, word_sb]
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


class WeightednCharGramOverlapFeature(Feature):

    def __init__(self, type='char', **kwargs):
        super(WeightednCharGramOverlapFeature, self).__init__(**kwargs)
        self.type = type
        self.feature_name += '-{}'.format(self.type)

    def word_to_char(self, word_list, pos_list):
        """
        Translate word_list to char_list
        """

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

    def extract(self, train_instance):
        pos_sa, pos_sb = train_instance.get_list('pos')
        word_sa, word_sb =train_instance.get_list('word')
        utils.word2char(word_sa)

        weighted_inter = 0
        weighted_seq1 = 0

        for pos, word in zip(pos_sa, word_sa):
            if pos[0] == 'N':
                score = 3.
            else:
                score = 0.
            if word in word_sb:
                weighted_inter += score
            weighted_seq1 += score

        weighted_overlap = weighted_inter / weighted_seq1 if weighted_seq1 != 0 else 0.0

        features = [weighted_overlap]
        infos = [word_sa, word_sb]
        return features, infos