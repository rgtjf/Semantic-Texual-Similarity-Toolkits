from stst.features.features import Feature
from stst import dict_utils, utils


class BOWFeature(Feature):
    def __init__(self, stopwords=True, **kwargs):
        super(BOWFeature, self).__init__(**kwargs)
        self.stopwords = stopwords
        self.feature_name = self.feature_name + '-%s' % (stopwords)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            lemma_sa, lemma_sb = train_instance.get_word(type='lemma', stopwords=self.stopwords, lower=True)
            seqs.append(lemma_sa)
            seqs.append(lemma_sb)

        self.idf_weight = utils.idf_calculator(seqs)

    def extract(self, train_instance):
        sa, sb = train_instance.get_word(type='lemma', stopwords=self.stopwords, lower=True)
        features, infos = utils.sentence_vectorize_features(sa, sb, self.idf_weight)
        return features, infos


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
