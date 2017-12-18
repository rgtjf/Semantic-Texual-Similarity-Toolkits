# coding: utf8
from __future__ import print_function

from collections import Counter

from stst.modules.features import Feature
from stst import utils
from stst.data import dict_utils


class nGramOverlapFeature(Feature):

    def __init__(self, type, **kwargs):
        """
        Args:
            type: word or lemma
        """
        super(nGramOverlapFeature, self).__init__(**kwargs)
        self.type = type
        self.feature_name += '-%s'%type

    def extract(self, train_instance):
        sa, sb = train_instance.get_word(type=self.type, stopwords=True)
        unigram_overlap = utils.ngram_match(sa, sb, 1)
        bigram_overlap = utils.ngram_match(sa, sb, 2)
        trigram_overlap = utils.ngram_match(sa, sb, 3)
        features = [unigram_overlap, bigram_overlap, trigram_overlap]
        infos = [sa, sb]
        return features, infos


class nCharGramOverlapFeature(Feature):
    """nCharGramOverlapFeature"""

    def __init__(self, stopwords, **kwargs):
        super(nCharGramOverlapFeature, self).__init__(**kwargs)
        self.stopwords = stopwords
        self.feature_name += '-%s'%stopwords

    def extract(self, train_instance):
        char_sa, char_sb = train_instance.get_char()
        bigram_overlap = utils.ngram_match(char_sa, char_sb, 2)
        trigram_overlap = utils.ngram_match(char_sa, char_sb, 3)
        four_gram_overlap = utils.ngram_match(char_sa, char_sb, 4)
        five_gram_overlap = utils.ngram_match(char_sa, char_sb, 5)
        features = [bigram_overlap, trigram_overlap, four_gram_overlap, five_gram_overlap]
        infos = [char_sa, char_sb]
        return features, infos


class nGramOverlapBeforeStopwordsFeature(Feature):
    """Word is lower, not remove stopwords"""

    def __init__(self, type, **kwargs):
        super(nGramOverlapBeforeStopwordsFeature, self).__init__(**kwargs)
        self.type = type
        self.feature_name += '-%s'%type

    def extract(self, train_instance):
        sa, sb = train_instance.get_word(type=self.type, lower=True)
        unigram_overlap, unigram_info = self.ngram_match_remove_stopwords(sa, sb, 1)
        bigram_overlap, bigram_info = self.ngram_match_remove_stopwords(sa, sb, 2)
        trigram_overlap, trigram_info = self.ngram_match_remove_stopwords(sa, sb, 3)
        features = [unigram_overlap, bigram_overlap, trigram_overlap]
        infos = [unigram_info, bigram_info, trigram_info]
        return features, infos

    @staticmethod
    def ngram_match_remove_stopwords(sa, sb, n):
        nga = utils.make_ngram(sa, n)
        ngb = utils.make_ngram(sb, n)

        stopwords = dict_utils.DictLoader().load_dict('stopwords')

        new_nga = []
        for ng in nga:
            new_ng = []
            for x in ng:
                if x not in stopwords:
                    new_ng.append(x)
            new_ng = tuple(new_ng)
            if new_ng != ():
                new_nga.append(new_ng)

        new_ngb = []
        for ng in ngb:
            new_ng = []
            for x in ng:
                if x.lower() not in stopwords:
                    new_ng.append(x)
            new_ng = tuple(new_ng)
            if new_ng != ():
                new_ngb.append(new_ng)

        f1 = utils.overlap_f1(new_nga, new_ngb)
        info = [new_nga, new_ngb]
        return f1, info


class WeightednGramOverlapFeature(Feature):
    """Word is lower, not remove stopwords"""

    def __init__(self, type, **kwargs):
        super(WeightednGramOverlapFeature, self).__init__(**kwargs)
        self.type=type
        self.feature_name += '-%s' % type

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            sa, sb = train_instance.get_word(type=self.type, lower=True)
            seqs.append(sa)
            seqs.append(sb)
        self.idf_weight = utils.idf_calculator(seqs)

    def extract(self, train_instance):
        sa, sb = train_instance.get_word(type=self.type, lower=True)
        unigram_overlap, unigram_info = self.weighted_ngram_match(sa, sb, 1, self.idf_weight)
        bigram_overlap, bigram_info = self.weighted_ngram_match(sa, sb, 2, self.idf_weight)
        trigram_overlap, trigram_info = self.weighted_ngram_match(sa, sb, 3, self.idf_weight)
        features = [unigram_overlap, bigram_overlap, trigram_overlap]
        infos = [sa, sb]
        return features, infos

    @staticmethod
    def weighted_ngram_match(sa, sb, n, idf_weight):
        """weighted_ngram_match
        """
        nga = utils.make_ngram(sa, n)
        ngb = utils.make_ngram(sb, n)
        min_idf_weight = min(idf_weight.values())

        def calc_ngram_idf(ngram):
            res = 0.0
            for ng in ngram:
                res += idf_weight.get(ng, min_idf_weight)
            return res

        idf_sa, idf_sb = 0.0, 0.0
        for ng in nga:
            idf_sa += calc_ngram_idf(ng)
        for ng in ngb:
            idf_sb += calc_ngram_idf(ng)

        matches = 0
        c1 = Counter(nga)
        info = []
        for ng in ngb:
            if c1[ng] > 0:
                c1[ng] -= 1
                matches += calc_ngram_idf(ng)
                info.append(ng)
        p, r, f1 = 0., 0., 1.
        if idf_sa > 0 and idf_sb > 0:
            p = matches / float(idf_sa)
            r = matches / float(idf_sb)
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
        return f1, info
