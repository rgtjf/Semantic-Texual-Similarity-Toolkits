# coding: utf8
from __future__ import print_function
from __future__ import unicode_literals

from stst.features.features import Feature
from stst import dict_utils, utils


class BOWFeature(Feature):

    def __init__(self, type='word', **kwargs):
        super(BOWFeature, self).__init__(**kwargs)
        self.type = type
        self.feature_name = self.feature_name + '-%s' % (type)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            lemma_sa, lemma_sb = train_instance.get_pair(type=self.type)
            seqs.append(lemma_sa)
            seqs.append(lemma_sb)

        self.idf_weight = utils.idf_calculator(seqs)
        self.vocab = utils.word2index(self.idf_weight)

    def extract(self, train_instance):
        sa, sb = train_instance.get_pair(type=self.type)
        vec1 = utils.vectorize(sa, self.idf_weight, self.vocab, convey='idf')
        vec2 = utils.vectorize(sb, self.idf_weight, self.vocab, convey='idf')
        features, info = [1-utils.cosine_distance(vec1, vec2, norm=True)], 'cosine'
        return features, info


class WeightedBOWFeature(Feature):

    def __init__(self, type='word', **kwargs):
        super(WeightedBOWFeature, self).__init__(**kwargs)
        self.type = type
        self.feature_name = self.feature_name + '-%s' % (type)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            word_sa, word_sb = train_instance.get_pair(type=self.type)
            seqs.append(word_sa)
            seqs.append(word_sb)

        self.idf_weight = utils.idf_calculator(seqs)
        self.vocab = utils.word2index(self.idf_weight)

    def extract(self, train_instance):
        sa, sb = train_instance.get_pair(type=self.type)
        vec1 = utils.vectorize(sa, self.idf_weight, self.vocab, convey='idf')
        vec2 = utils.vectorize(sb, self.idf_weight, self.vocab, convey='idf')

        pos_sa, pos_sb = train_instance.get_list('pos')
        word_sa, word_sb = train_instance.get_list('word')

        for pos, word in zip(pos_sa, word_sa):
            index = self.vocab[word]
            if pos[0] == 'N':
                vec1[index] *= 3

        for pos, word in zip(pos_sb, word_sb):
            index = self.vocab[word]
            if pos[0] == 'N':
                vec2[index] *= 3

        features, info = [1-utils.cosine_distance(vec1, vec2, norm=True)], 'cosine'
        return features, info

class BOWCountFeature(Feature):

    def __init__(self, type='word', **kwargs):
        super(BOWCountFeature, self).__init__(**kwargs)
        self.type = type
        self.feature_name = self.feature_name + '-%s' % (type)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            lemma_sa, lemma_sb = train_instance.get_pair(type=self.type)
            seqs.append(lemma_sa)
            seqs.append(lemma_sb)

        self.idf_weight = utils.idf_calculator(seqs)
        self.vocab = utils.word2index(self.idf_weight)

    def extract(self, train_instance):
        sa, sb = train_instance.get_pair(type=self.type)
        vec1 = utils.vectorize(sa, self.idf_weight, self.vocab, convey='count')
        vec2 = utils.vectorize(sb, self.idf_weight, self.vocab, convey='count')
        features, info = [1-utils.cosine_distance(vec1, vec2, norm=True)], 'cosine'
        return features, info


class BOWGlobalFeature(Feature):
    """
    Idf use all sentences other than these in one corpus(file),
    however BOWFeature is superior than this BOWGlobalFeature
    """
    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            lemma_sa, lemma_sb = train_instance.get_pair(type='word')
            seqs.append(lemma_sa)
            seqs.append(lemma_sb)

        local_idf_weight = utils.idf_calculator(seqs)
        self.vocab = utils.word2index(local_idf_weight)
        self.vocab['<UNK>'] = len(self.vocab)

        self.idf_weight = dict_utils.DictLoader().load_dict('idf_weight', 'idf.weight', sep=chr(1))
        for word in self.vocab:
            if word in self.idf_weight:
                print(word, self.idf_weight[word])
            else:
                print(word, 'unk')

    def extract(self, train_instance):
        sa, sb = train_instance.get_pair(type='word')
        vec1 = utils.vectorize(sa, self.idf_weight, self.vocab, convey='idf', unk_word='<UNK>')
        vec2 = utils.vectorize(sb, self.idf_weight, self.vocab, convey='idf', unk_word='<UNK>')
        features, info = [1 - utils.cosine_distance(vec1, vec2, norm=True)], 'cosine'
        return features, info



class DependencyFeature(Feature):

    def __init__(self, type='word_dep', convey='idf',**kwargs):
        super(DependencyFeature, self).__init__(**kwargs)
        self.type = type
        self.convey = convey
        self.feature_name += '-{}-{}'.format(type, convey)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            lemma_sa, lemma_sb = train_instance.get_pair(type=self.type)
            seqs.append(lemma_sa)
            seqs.append(lemma_sb)

        self.idf_weight = utils.idf_calculator(seqs)
        self.vocab = utils.word2index(self.idf_weight)

    def extract(self, train_instance):
        sa, sb = train_instance.get_pair(type=self.type)
        vec1 = utils.vectorize(sa, self.idf_weight, self.vocab, convey='idf')
        vec2 = utils.vectorize(sb, self.idf_weight, self.vocab, convey='idf')
        features, info = [1-utils.cosine_distance(vec1, vec2, norm=True)], 'cosine'
        return features, info