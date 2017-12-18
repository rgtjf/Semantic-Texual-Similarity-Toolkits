# coding: utf8

from stst.modules.features import Feature
from stst import utils


class SequenceFeature(Feature):

    def extract(self, train_instance):
        sa, sb = train_instance.get_word(type='lemma', stopwords=True, lower=True)

        la, lb = len(sa), len(sb)
        l = min(la, lb)

        features = []
        feature, info = utils.sentence_sequence_features(sa, sb)
        features += feature

        infos = [sa, sb]
        return features, infos


class SentenceFeature(Feature):
    def extract(self, train_instance):
        sa, sb = train_instance.get_preprocess()
        # sa, sb = train_instance.get_word(type='lemma', stopwords=True, lower=True)

        la, lb = len(sa), len(sb)
        l = min(la, lb)

        features = []
        feature, info = utils.sentence_sequence_features(sa, sb)
        features += feature

        feature, info = utils.sentence_match_features(sa, sb)
        features += feature

        bow = utils.idf_calculator([sa, sb])
        feature, info = utils.sentence_vectorize_features(sa, sb, bow, convey='count')
        features += feature
        infos = [sa, sb]
        return features, infos


class SequenceBakFeature(Feature):
    def extract(self, train_instance):
        sa, sb = train_instance.get_preprocess()
        # sa, sb = train_instance.get_word(type='lemma', stopwords=True, lower=True)

        la, lb = len(sa), len(sb)
        l = min(la, lb)

        features = []
        feature, info = utils.sentence_sequence_features(sa, sb)
        features += feature

        feature, info = utils.sentence_match_features(sa, sb)
        features += feature

        bow = utils.idf_calculator([sa, sb])
        feature, info = utils.sentence_vectorize_features(sa, sb, bow, convey='count')
        features += feature
        infos = [sa, sb]
        return features, infos
