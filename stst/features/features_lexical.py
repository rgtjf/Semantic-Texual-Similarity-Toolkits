from features import Feature
from .. import utils
from ..lib.sentence_similarity.short_sentence_similarity import *


class ShortSentenceFeature(Feature):
    def extract(self, train_instance):
        sa, sb = train_instance.get_word(type='word')
        features = [semantic_similarity(sa, sb, True), semantic_similarity(sa, sb, False)]
        infos = []
        return features, infos


class LexicalFeature(Feature):
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
