# coding: utf8
from __future__ import print_function

from collections import Counter

import dict_utils
import nltk
import numpy as np
from nltk.corpus import brown
from nltk.corpus import wordnet

from features import Feature


class nLemmaGramOverlapFeature(Feature):
    def _ngram_match(self, sa, sb, n):

        def _make_ngrams(sent, n):
            rez = [sent[i:(-n + i + 1)] for i in range(n - 1)]
            rez.append(sent[n - 1:])
            return zip(*rez)

        nga = _make_ngrams(sa, n)
        ngb = _make_ngrams(sb, n)

        matches = 0
        c1 = Counter(nga)
        info = []
        for ng in ngb:
            if c1[ng] > 0:
                c1[ng] -= 1
                matches += 1
                info.append(ng)
        p = 0.
        r = 0.
        f1 = 1.
        if len(nga) > 0 and len(ngb) > 0:
            p = matches / float(len(nga))
            r = matches / float(len(ngb))
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
        return f1, info

    def extract(self, train_instance):
        lemma_sa, lemma_sb = train_instance.get_word(type='lemma', stopwords=True)
        features, infos = self.core_extract(lemma_sa, lemma_sb)
        return features, infos

    def core_extract(self, sa, sb):
        unigram_overlap, unigram_info = self._ngram_match(sa, sb, 1)
        bigram_overlap, bigram_info = self._ngram_match(sa, sb, 2)
        trigram_overlap, trigram_info = self._ngram_match(sa, sb, 3)
        return [unigram_overlap, bigram_overlap, trigram_overlap], [unigram_info, bigram_info, trigram_info]


class nWordGramOverlapFeature(Feature):
    def _ngram_match(self, sa, sb, n):

        def _make_ngrams(sent, n):
            rez = [sent[i:(-n + i + 1)] for i in range(n - 1)]
            rez.append(sent[n - 1:])
            return zip(*rez)

        nga = _make_ngrams(sa, n)
        ngb = _make_ngrams(sb, n)

        matches = 0
        c1 = Counter(nga)
        info = []
        for ng in ngb:
            if c1[ng] > 0:
                c1[ng] -= 1
                matches += 1
                info.append(ng)
        p = 0.
        r = 0.
        f1 = 1.
        if len(nga) > 0 and len(ngb) > 0:
            p = matches / float(len(nga))
            r = matches / float(len(ngb))
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
        return f1, info

    def extract(self, train_instance):
        word_sa, word_sb = train_instance.get_word(type='word', stopwords=True)
        features, infos = self.core_extract(word_sa, word_sb)
        return features, infos

    def core_extract(self, sa, sb):
        unigram_overlap, unigram_info = self._ngram_match(sa, sb, 1)
        bigram_overlap, bigram_info = self._ngram_match(sa, sb, 2)
        trigram_overlap, trigram_info = self._ngram_match(sa, sb, 3)
        return [unigram_overlap, bigram_overlap, trigram_overlap], [unigram_info, bigram_info, trigram_info]


class nCharGramOverlapFeature(Feature):
    def _ngram_match(self, sa, sb, n):

        def _make_ngrams(sent, n):
            rez = [sent[i:(-n + i + 1)] for i in range(n - 1)]
            rez.append(sent[n - 1:])
            return zip(*rez)

        nga = _make_ngrams(sa, n)
        ngb = _make_ngrams(sb, n)

        matches = 0
        c1 = Counter(nga)
        info = []
        for ng in ngb:
            if c1[ng] > 0:
                c1[ng] -= 1
                matches += 1
                info.append(ng)
        p = 0.
        r = 0.
        f1 = 1.
        if len(nga) > 0 and len(ngb) > 0:
            p = matches / float(len(nga))
            r = matches / float(len(ngb))
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
        return f1, info

    def extract(self, train_instance):
        char_sa, char_sb = train_instance.get_char()
        features, infos = self.core_extract(char_sa, char_sb)
        infos = [[''.join(x) for x in info] for info in infos]
        return features, infos

    def core_extract(self, sa, sb):
        bigram_overlap, bigram_info = self._ngram_match(sa, sb, 2)
        trigram_overlap, trigram_info = self._ngram_match(sa, sb, 3)
        four_gram_overlap, four_gram_info = self._ngram_match(sa, sb, 4)
        five_gram_overlap, five_gram_info = self._ngram_match(sa, sb, 5)
        return [bigram_overlap, trigram_overlap, four_gram_overlap, five_gram_overlap], [bigram_info, trigram_info,
                                                                                         four_gram_info, five_gram_info]


class nWordGramOverlapBeforeStopwordsFeature(Feature):
    """
    Word is lower, not remove stopwords
    """

    def _ngram_match(self, sa, sb, n):

        def _make_ngrams(sent, n):
            rez = [sent[i:(-n + i + 1)] for i in range(n - 1)]
            rez.append(sent[n - 1:])
            return zip(*rez)

        nga = _make_ngrams(sa, n)
        ngb = _make_ngrams(sb, n)

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

        matches = 0
        c1 = Counter(nga)
        info = []
        for ng in ngb:
            if c1[ng] > 0:
                c1[ng] -= 1
                matches += 1
                info.append(ng)
        p = 0.
        r = 0.
        f1 = 1.
        if len(nga) > 0 and len(ngb) > 0:
            p = matches / float(len(nga))
            r = matches / float(len(ngb))
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
        return f1, info

    def extract(self, train_instance):
        lemma_sa, lemma_sb = train_instance.get_word(type='word', lower=True)
        features, infos = self.core_extract(lemma_sa, lemma_sb)
        return features, infos

    def core_extract(self, sa, sb):
        unigram_overlap, unigram_info = self._ngram_match(sa, sb, 1)
        bigram_overlap, bigram_info = self._ngram_match(sa, sb, 2)
        trigram_overlap, trigram_info = self._ngram_match(sa, sb, 3)
        return [unigram_overlap, bigram_overlap, trigram_overlap], [unigram_info, bigram_info, trigram_info]


class nLemmaGramOverlapBeforeStopwordsFeature(Feature):
    def _ngram_match(self, sa, sb, n):

        def _make_ngrams(sent, n):
            rez = [sent[i:(-n + i + 1)] for i in range(n - 1)]
            rez.append(sent[n - 1:])
            return zip(*rez)

        nga = _make_ngrams(sa, n)
        ngb = _make_ngrams(sb, n)

        stopwords = dict_utils.DictLoader().load_dict('stopwords')

        new_nga = []
        for ng in nga:
            new_ng = []
            for x in ng:
                if x.lower() not in stopwords:
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

        # print(new_nga, new_ngb)

        nga = new_nga
        ngb = new_ngb

        matches = 0
        c1 = Counter(nga)
        info = []
        for ng in ngb:
            if c1[ng] > 0:
                c1[ng] -= 1
                matches += 1
                info.append(ng)
        p = 0.
        r = 0.
        f1 = 1.
        if len(nga) > 0 and len(ngb) > 0:
            p = matches / float(len(nga))
            r = matches / float(len(ngb))
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
        return f1, info

    def extract(self, train_instance):
        lemma_sa, lemma_sb = train_instance.get_word(type='lemma')
        features, infos = self.core_extract(lemma_sa, lemma_sb)
        return features, infos

    def core_extract(self, sa, sb):
        unigram_overlap, unigram_info = self._ngram_match(sa, sb, 1)
        bigram_overlap, bigram_info = self._ngram_match(sa, sb, 2)
        trigram_overlap, trigram_info = self._ngram_match(sa, sb, 3)
        return [unigram_overlap, bigram_overlap, trigram_overlap], [unigram_info, bigram_info, trigram_info]


class nCharGramNonStopwordsOverlapFeature(Feature):
    def _ngram_match(self, sa, sb, n):

        def _make_ngrams(sent, n):
            rez = [sent[i:(-n + i + 1)] for i in range(n - 1)]
            rez.append(sent[n - 1:])
            return zip(*rez)

        nga = _make_ngrams(sa, n)
        ngb = _make_ngrams(sb, n)

        matches = 0
        c1 = Counter(nga)
        info = []
        for ng in ngb:
            if c1[ng] > 0:
                c1[ng] -= 1
                matches += 1
                info.append(ng)
        p = 0.
        r = 0.
        f1 = 1.
        if len(nga) > 0 and len(ngb) > 0:
            p = matches / float(len(nga))
            r = matches / float(len(ngb))
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
        return f1, info

    def extract(self, train_instance):
        char_sa, char_sb = train_instance.get_char(stopwords=True)
        features, infos = self.core_extract(char_sa, char_sb)
        infos = [[''.join(x) for x in info] for info in infos]
        return features, infos

    def core_extract(self, sa, sb):
        bigram_overlap, bigram_info = self._ngram_match(sa, sb, 2)
        trigram_overlap, trigram_info = self._ngram_match(sa, sb, 3)
        four_gram_overlap, four_gram_info = self._ngram_match(sa, sb, 4)
        five_gram_overlap, five_gram_info = self._ngram_match(sa, sb, 5)
        return [bigram_overlap, trigram_overlap, four_gram_overlap, five_gram_overlap], [bigram_info, trigram_info,
                                                                                         four_gram_info, five_gram_info]
