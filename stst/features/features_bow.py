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
    def __init__(self, **kwargs):
        super(BOWGlobalFeature, self).__init__(**kwargs)


    def extract(self, train_instance):
        idf_weight = dict_utils.DictLoader().load_dict('global_idf')
        sa, sb = train_instance.get_word(type='lemma', stopwords=True, lower=True)
        features, infos = utils.sentence_vectorize_features(sa, sb, idf_weight)

        return features, infos
