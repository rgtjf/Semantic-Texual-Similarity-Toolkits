from features import Feature
from ..lib.sentence_similarity.short_sentence_similarity import *
from .. import utils


class SequenceFeature(Feature):
    def extract(self, train_instance):
        sa, sb = train_instance.get_preprocess()
        # sa, sb = train_instance.get_word(type='lemma', stopwords=True, lower=True)

        features = []
        feature, info = utils.sequence_edit_distance_features(sa, sb)
        features += feature

        infos = [sa, sb]
        return features, infos
