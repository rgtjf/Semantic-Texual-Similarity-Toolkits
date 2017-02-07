from features import Feature
from lib.short_sentence_similarity import *
import utils, dict_utils


class ShortSentenceFeature(Feature):
    def extract(self, train_instance):
        sa, sb = train_instance.get_word(type='word')
        features = [semantic_similarity(sa, sb, True), semantic_similarity(sa, sb, False)]
        infos = []
        return features, infos


class SequenceFeature(Feature):
    def extract(self, train_instance):
        sa, sb = train_instance.get_preprocess()
        # sa, sb = train_instance.get_word(type='lemma', stopwords=True, lower=True)

        la, lb = len(sa), len(sb)
        l = min(la, lb)

        features = []
        feature, info = utils.sequence_edit_distance_features(sa, sb)
        features += feature

        feature, info = utils.sequence_match_features(sa, sb)
        features += feature

        bow = utils.IDFCalculator([sa, sb])
        feature, info = utils.sequence_vector_features(sa, sb, bow, convey='count')
        features += feature

        infos = [sa, sb]
        return features, infos


class BOWFeature(Feature):
    def __init__(self, stopwords=True, **karwgs):
        super(BOWFeature, self).__init__(**karwgs)
        self.stopwords = stopwords
        self.feature_name = self.feature_name + '-%s' % (stopwords)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            lemma_sa, lemma_sb = train_instance.get_word(type='lemma', stopwords=self.stopwords, lower=True)
            seqs.append(lemma_sa)
            seqs.append(lemma_sb)

        self.idf_weight = utils.IDFCalculator(seqs)

    def extract(self, train_instance):
        sa, sb = train_instance.get_word(type='lemma', stopwords=self.stopwords, lower=True)

        # rs = [
        #     # utils.sequence_match_features(sa, sb),
        #     utils.sequence_vector_features(sa, sb, self.idf_weight)
        # ]
        # features = [val for r in rs for val in r[0]]
        # infos = [val for r in rs for val in r[1]]

        features, infos = utils.sequence_vector_features(sa, sb, self.idf_weight)

        return features, infos

class BOWSmallFeature(Feature):
    def __init__(self, stopwords=True, **karwgs):
        super(BOWSmallFeature, self).__init__(**karwgs)
        self.stopwords = stopwords
        self.feature_name = self.feature_name + '-%s' % (stopwords)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            lemma_sa, lemma_sb = train_instance.get_word(type='lemma', stopwords=self.stopwords, lower=True)
            seqs.append(lemma_sa)
            seqs.append(lemma_sb)

        self.idf_weight = utils.IDFCalculator(seqs)

    def extract(self, train_instance):
        sa, sb = train_instance.get_word(type='lemma', stopwords=self.stopwords, lower=True)

        features, infos = utils.sequence_small_vector_features(sa, sb, self.idf_weight)

        return features, infos


class BOWGlobalFeature(Feature):
    def __init__(self, **karwgs):
        super(BOWGlobalFeature, self).__init__(**karwgs)


    def extract(self, train_instance):

        idf_weight = dict_utils.DictLoader().load_dict('global_idf')
        sa, sb = train_instance.get_word(type='lemma', stopwords=True, lower=True)

        # features, infos = utils.sequence_small_vector_features(sa, sb, idf_weight)

        features, infos = utils.sequence_vector_features(sa, sb, idf_weight)

        return features, infos

