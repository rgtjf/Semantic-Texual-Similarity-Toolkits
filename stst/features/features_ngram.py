# coding: utf8
from features import Feature
from collections import Counter
import dict_utils


def match_features(match, len_sa, len_sb):
    p, r, f1 = 0., 0., 1.
    if len_sa > 0 and len_sb > 0:
        p = match / float(len_sa)
        r = match / float(len_sb)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return [f1] # [p, r, f1]


def ngram_match(sa, sb, n):
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
    features = match_features(matches, len(nga), len(ngb))
    return features, info


def ngram_match_stopwords(sa, sb, n):
    def _make_ngrams(sent, n):
        rez = [sent[i:(-n + i + 1)] for i in range(n - 1)]
        rez.append(sent[n - 1:])
        return zip(*rez)

    nga = _make_ngrams(sa, n)
    ngb = _make_ngrams(sb, n)

    stopwords = dict_utils.DictLoader().load_dict('stopwords')
    idf_weight = dict_utils.DictLoader().load_dict('idf')

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

    def calc_ngram_idf(ngram):
        res = 0.0
        for ng in ngram:
            res += idf_weight.get(ng, 10.0)
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
    features = match_features(matches, idf_sa, idf_sb)
    return features, info


class nLemmaGramOverlapMatchFeature(Feature):
    def extract(self, train_instance):
        lemma_sa, lemma_sb = train_instance.get_word(type='lemma', stopwords=True)
        features, infos = self.core_extract(lemma_sa, lemma_sb)
        return features, infos

    def core_extract(self, sa, sb):
        unigram_overlap, unigram_info = ngram_match(sa, sb, 1)
        bigram_overlap, bigram_info = ngram_match(sa, sb, 2)
        trigram_overlap, trigram_info = ngram_match(sa, sb, 3)
        return unigram_overlap + bigram_overlap + trigram_overlap, [unigram_info, bigram_info, trigram_info]


class nWordGramOverlapMatchFeature(Feature):
    def extract(self, train_instance):
        word_sa, word_sb = train_instance.get_word(type='word', stopwords=True)
        features, infos = self.core_extract(word_sa, word_sb)
        return features, infos

    def core_extract(self, sa, sb):
        unigram_overlap, unigram_info = ngram_match(sa, sb, 1)
        bigram_overlap, bigram_info = ngram_match(sa, sb, 2)
        trigram_overlap, trigram_info = ngram_match(sa, sb, 3)
        return unigram_overlap + bigram_overlap + trigram_overlap, [unigram_info, bigram_info, trigram_info]


class nCharGramOverlapMatchFeature(Feature):
    def extract(self, train_instance):
        char_sa, char_sb = train_instance.get_char()
        features, infos = self.core_extract(char_sa, char_sb)
        infos = [[''.join(x) for x in info] for info in infos]
        return features, infos

    def core_extract(self, sa, sb):
        bigram_overlap, bigram_info = ngram_match(sa, sb, 2)
        trigram_overlap, trigram_info = ngram_match(sa, sb, 3)
        four_gram_overlap, four_gram_info = ngram_match(sa, sb, 4)
        five_gram_overlap, five_gram_info = ngram_match(sa, sb, 5)
        return bigram_overlap + trigram_overlap + four_gram_overlap + five_gram_overlap, \
               [bigram_info, trigram_info, four_gram_info, five_gram_info]


class nWordGramOverlapBeforeStopwordsMatchFeature(Feature):
    """
    Word is lower, not remove stopwords
    """

    def extract(self, train_instance):
        lemma_sa, lemma_sb = train_instance.get_word(type='word', lower=True)
        features, infos = self.core_extract(lemma_sa, lemma_sb)
        return features, infos

    def core_extract(self, sa, sb):
        unigram_overlap, unigram_info = ngram_match_stopwords(sa, sb, 1)
        bigram_overlap, bigram_info = ngram_match_stopwords(sa, sb, 2)
        trigram_overlap, trigram_info = ngram_match_stopwords(sa, sb, 3)
        return unigram_overlap + bigram_overlap + trigram_overlap, [unigram_info, bigram_info, trigram_info]


class nLemmaGramOverlapBeforeStopwordsMatchFeature(Feature):
    def extract(self, train_instance):
        lemma_sa, lemma_sb = train_instance.get_word(type='lemma')
        features, infos = self.core_extract(lemma_sa, lemma_sb)
        return features, infos

    def core_extract(self, sa, sb):
        unigram_overlap, unigram_info = ngram_match_stopwords(sa, sb, 1)
        bigram_overlap, bigram_info = ngram_match_stopwords(sa, sb, 2)
        trigram_overlap, trigram_info = ngram_match_stopwords(sa, sb, 3)
        return unigram_overlap + bigram_overlap + trigram_overlap, [unigram_info, bigram_info, trigram_info]


class nCharGramNonStopwordsOverlapMatchFeature(Feature):
    def extract(self, train_instance):
        char_sa, char_sb = train_instance.get_char(stopwords=True)
        features, infos = self.core_extract(char_sa, char_sb)
        infos = [[''.join(x) for x in info] for info in infos]
        return features, infos

    def core_extract(self, sa, sb):
        bigram_overlap, bigram_info = ngram_match(sa, sb, 2)
        trigram_overlap, trigram_info = ngram_match(sa, sb, 3)
        four_gram_overlap, four_gram_info = ngram_match(sa, sb, 4)
        five_gram_overlap, five_gram_info = ngram_match(sa, sb, 5)
        return bigram_overlap + trigram_overlap + four_gram_overlap + five_gram_overlap, \
               [bigram_info, trigram_info, four_gram_info, five_gram_info]


