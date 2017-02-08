from features import Feature
from stst import utils


class DependencyGramFeature(Feature):
    def __init__(self, convey='count', **karwgs):
        super(DependencyGramFeature, self).__init__(**karwgs)
        self.convey = convey
        self.feature_name = self.feature_name + '-%s' % (convey)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            dep_sa, dep_sb = train_instance.get_dependency()
            dep_sa = [(dep[1], dep[2]) for dep in dep_sa]
            dep_sb = [(dep[1], dep[2]) for dep in dep_sb]
            seqs.append(dep_sa)
            seqs.append(dep_sb)
        self.idf_weight = utils.IDFCalculator(seqs)

    def extract(self, train_instance):
        dep_sa, dep_sb = train_instance.get_dependency()
        dep_sa = [(dep[1], dep[2]) for dep in dep_sa]
        dep_sb = [(dep[1], dep[2]) for dep in dep_sb]

        features = []
        feature, info = utils.sequence_match_features(dep_sa, dep_sb)
        features += feature

        feature, info = utils.sequence_vector_features(dep_sa, dep_sb, self.idf_weight, convey=self.convey)
        features += feature

        infos = [dep_sa, dep_sb]
        return features, infos


class DependencyRelationFeature(Feature):
    def __init__(self, convey='count', **karwgs):
        super(DependencyRelationFeature, self).__init__(**karwgs)
        self.convey = convey
        self.feature_name = self.feature_name + '-%s' % (convey)
        # self.feature_name = self.feature_name + '-%s' % (stopwords)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            dep_sa, dep_sb = train_instance.get_dependency()
            seqs.append(dep_sa)
            seqs.append(dep_sb)
        self.idf_weight = utils.IDFCalculator(seqs)

    def extract(self, train_instance):
        dep_sa, dep_sb = train_instance.get_dependency()

        features = []
        feature, info = utils.sequence_match_features(dep_sa, dep_sb)
        features += feature

        feature, info = utils.sequence_vector_features(dep_sa, dep_sb, self.idf_weight, convey=self.convey)
        features += feature

        infos = [dep_sa, dep_sb]
        return features, infos
