# coding: utf8
from __future__ import print_function
import codecs, utils
from collections import Counter
import config, utils
import json, pyprind
import dict_utils
import numpy as np
import nltk
import os
import math
from nltk.corpus import brown
from nltk.corpus import wordnet
from features import Feature
import lib.kernel.vector_kernel as vk


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


class NegationPenaltyAlignmentFeature(Feature):
    def extract(self, train_instance):
        pass


class NegativeFeature(Feature):
    def extract(self, train_instance):
        negation = dict_utils.DictLoader().load_dict('negation_terms')
        lemma_sa, lemma_sb = train_instance.get_word(type='lemma', lower=True)
        na = sum([1 if w in negation else 0 for w in lemma_sa])
        nb = sum([1 if w in negation else 0 for w in lemma_sb])

        features = [(na - nb) % 2]
        infos = [na, nb]
        return features, infos


class EnNegativeFeature(Feature):
    def __init__(self, penalty, **kwargs):
        super(EnNegativeFeature, self).__init__(**kwargs)
        self.load = False
        self.penalty = penalty

    def extract(self, train_instance):
        negation = dict_utils.DictLoader().load_dict('negation_terms')
        lemma_sa, lemma_sb = train_instance.get_word(type='lemma', lower=True)
        na = sum([1 if w in negation else 0 for w in lemma_sa])
        nb = sum([1 if w in negation else 0 for w in lemma_sb])

        if (na - nb) % 2 != 0:
            score = self.penalty
        else:
            score = 0.0
        features = [score]
        infos = [na, nb]
        return features, infos


class LengthFeature(Feature):
    def extract(self, train_instance):
        """
        Extract features and info from one train instance
        """
        word_sa, word_sb = train_instance.get_word(type='word', stopwords=True)
        length_sa = len(word_sa)
        length_sb = len(word_sb)
        p = 1.0 * length_sa / length_sb if length_sb != 0 else 0.0
        r = 1.0 * length_sb / length_sa if length_sa != 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
        features = [f1, length_sa + length_sb]
        infos = [length_sa, length_sb]
        return features, infos


brown_ic = wordnet.ic(brown, False, 0.0)


class WordNetFeatures(Feature):
    def path_sim(self, word_with_pos1, word_with_pos2):
        sim_vec = []
        for (word1, pos1) in word_with_pos1:
            for (word2, pos2) in word_with_pos2:
                syn1, syn2, r = self.get_synset(word1, pos1, word2, pos2)
                if r == -1:
                    continue
                sim = syn1.path_similarity(syn2)
                if sim == -1 or sim == None:
                    sim_vec.append(0)
                else:
                    sim_vec.append(sim)
        if len(sim_vec) == 0:
            sim_vec = [0]
        return [np.max(sim_vec), np.min(sim_vec), np.mean(sim_vec)]

    def lch_sim(self, word_with_pos1, word_with_pos2):
        sim_vec = []
        for (word1, pos1) in word_with_pos1:
            for (word2, pos2) in word_with_pos2:
                syn1, syn2, r = self.get_synset(word1, pos1, word2, pos2)
                if r == -1 or not syn1.name().split('.')[1] == syn2.name().split('.')[1]:
                    continue
                try:
                    sim = syn1.lch_similarity(syn2)
                except:
                    sim = 1
                if sim == -1 or sim == None:
                    sim_vec.append(0)
                else:
                    sim_vec.append(sim)
        if len(sim_vec) == 0:
            sim_vec = [0]
        return [np.max(sim_vec), np.min(sim_vec), np.mean(sim_vec)]

    def wup_sim(self, word_with_pos1, word_with_pos2):
        sim_vec = []
        for (word1, pos1) in word_with_pos1:
            for (word2, pos2) in word_with_pos2:
                syn1, syn2, r = self.get_synset(word1, pos1, word2, pos2)
                if r == -1:
                    continue
                sim = syn1.wup_similarity(syn2)
                if sim == -1 or sim == None:
                    sim_vec.append(0)
                else:
                    sim_vec.append(sim)
        if len(sim_vec) == 0:
            sim_vec = [0]
        return [np.max(sim_vec), np.min(sim_vec), np.mean(sim_vec)]

    def jcn_sim(self, word_with_pos1, word_with_pos2, ic):
        sim_vec = []
        for (word1, pos1) in word_with_pos1:
            for (word2, pos2) in word_with_pos2:
                syn1, syn2, r = self.get_synset(word1, pos1, word2, pos2)
                if r == -1 or not syn1.name().split('.')[1] == syn2.name().split('.')[1]:
                    continue
                try:
                    sim = syn1.jcn_similarity(syn2, ic)
                except:
                    continue
                if sim == -1 or sim == None:
                    sim_vec.append(0)
                elif sim > 1.0 or sim < 0:
                    sim_vec.append(1.0)
                else:
                    sim_vec.append(sim)
        if len(sim_vec) == 0:
            sim_vec = [0]
        return [np.max(sim_vec), np.min(sim_vec), np.mean(sim_vec)]

    def lin_sim(self, word_with_pos1, word_with_pos2, ic):
        sim_vec = []
        for (word1, pos1) in word_with_pos1:
            for (word2, pos2) in word_with_pos2:
                syn1, syn2, r = self.get_synset(word1, pos1, word2, pos2)
                if r == -1 or not syn1.name().split('.')[1] == syn2.name().split('.')[1]:
                    continue
                try:
                    sim = syn1.lin_similarity(syn2, ic)
                except:
                    continue

                if sim == -1 or sim == None:
                    sim_vec.append(0)
                elif sim > 1.0 or sim < 0:
                    sim_vec.append(1.0)
                else:
                    sim_vec.append(sim)
        if len(sim_vec) == 0:
            sim_vec = [0]
        return [np.max(sim_vec), np.min(sim_vec), np.mean(sim_vec)]

    def get_synset(self, word1, pos1, word2, pos2):
        if not pos1 == '#' and not pos2 == '#':
            try:
                syn1 = wordnet.synset(word1 + '.' + pos1 + '.01')
                syn2 = wordnet.synset(word2 + '.' + pos2 + '.01')
            except nltk.corpus.reader.wordnet.WordNetError:
                # print('WordNetError')
                return None, None, -1
            return syn1, syn2, 1
        return None, None, -1

    def get_antonyms(self, word):
        synsets = wordnet.synsets(word)
        lemmas = []
        for synset in synsets:
            lemma = synset.lemmas()
            lemmas += lemma
        antonyms = []
        for lemma in lemmas:
            antonyms += lemma.antonyms()
        antonyms = [antonym.name for antonym in antonyms]
        return set(antonyms)

    def is_antonyms(self, word1, word2):
        anto = self.get_antonyms(word1)
        if word2 in anto:
            return True
        anto = self.get_antonyms(word2)
        if word1 in anto:
            return True
        return False

    def negation_detect(self, word_list1, word_list2, neg_list):
        flag1 = False
        flag2 = False
        for val in neg_list:
            if val in word_list1:
                flag1 = True
                break
        for val in neg_list:
            if val in word_list2:
                flag2 = True
                break
        if not flag1 == flag2:
            return 1
        elif flag1 == False:
            for word1 in word_list1:
                for word2 in word_list2:
                    if self.is_antonyms(word1, word2):
                        return 1
            lens = len(word_list1) if len(word_list1) < len(word_list2) else len(word_list2)
            st = 0
            for index in range(lens):
                st = index
                if not word_list1[index] == word_list2[index]:
                    break
            len1 = len(word_list1)
            len2 = len(word_list2)
            index = 1
            while index <= len1 and index <= len2 and word_list1[len1 - index] == word_list2[len2 - index]:
                index += 1
            if index > 1 and st > 0:
                return 0
            return -1
        else:
            return -1

    def extract(self, train_instance, ):
        # lemma_sa, lemma_sb = train_instance.get_word(type='lemma', stopwords=True, lower=True)
        pos_sa, pos_sb = train_instance.get_pos_tag()

        # neg_list = dict_loader().neg_list
        # feats.append(self.negation_detect(lemma_sa, lemma_sb, neg_list))


        features = []

        diff_sent_1_2 = [val for val in pos_sa if not val in pos_sb]
        diff_sent_2_1 = [val for val in pos_sb if not val in pos_sa]

        diff_sent_1_2 = pos_sa
        diff_sent_2_1 = pos_sb

        if not diff_sent_1_2 == [] and not diff_sent_2_1 == []:
            features += self.path_sim(diff_sent_1_2, diff_sent_2_1)
            features += self.lch_sim(diff_sent_1_2, diff_sent_2_1)
            features += self.wup_sim(diff_sent_1_2, diff_sent_2_1)
            features += self.jcn_sim(diff_sent_1_2, diff_sent_2_1, brown_ic)
        elif diff_sent_1_2 == [] and diff_sent_2_1 == []:
            features += [1] * 12
        else:
            features += [0.5] * 12

        infos = [diff_sent_1_2, diff_sent_2_1]

        return features, infos


if __name__ == '__main__':

    def _make_ngrams(l, n):
        rez = [l[i:(-n + i + 1)] for i in range(n - 1)]
        rez.append(l[n - 1:])
        return zip(*rez)


    def ngram_match(sa, sb, n):
        nga = _make_ngrams(sa, n)
        ngb = _make_ngrams(sb, n)
        matches = 0
        c1 = Counter(nga)
        for ng in ngb:
            if c1[ng] > 0:
                c1[ng] -= 1
                matches += 1
        p = 0.
        r = 0.
        f1 = 1.
        if len(nga) > 0 and len(ngb) > 0:
            p = matches / float(len(nga))
            r = matches / float(len(ngb))
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
        return f1


    sa = "in the interview publish on Saturday night , Guzman say he enter the drug business at 15 year old because there be no other way to survive ."
    sb = "in the interview , Guzman defend he work at the head of the world 's biggest drug trafficking organization , one blame for thousand of killing ."
    score = 0.44
    sa = sa.split()
    sb = sb.split()

    print(_make_ngrams(sa, 3))
    print(nLemmaGramOverlapFeature().core_extract(sa, sb))
