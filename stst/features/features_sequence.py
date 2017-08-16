from stst.features.features import Feature
from stst import utils


class SequenceFeature(Feature):

    def __init__(self, type='word', **kwargs):
        super(SequenceFeature, self).__init__(**kwargs)
        self.type = type

    def extract(self, train_instance):
        sa, sb = train_instance.get_pair(self.type)
        # ['prefix', 'suffix', 'longest_common_substring',
        #   'longest_common_sequence', 'levenshtein_disttance']
        features, infos = utils.sentence_sequence_features(sa, sb)
        features = features[2:]
        return features, infos


class SentenceFeature(Feature):

    def __init__(self, type='word', **kwargs):
        super(SentenceFeature, self).__init__(**kwargs)
        self.type = type
        self.feature_name += '-{}'.format(self.type)

    def extract(self, train_instance):
        sa, sb = train_instance.get_pair(self.type)

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
