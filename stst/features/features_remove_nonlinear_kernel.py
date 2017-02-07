import os, json
import pyprind
import utils
from features import Feature


class BOW6kFeature(Feature):
    def __init__(self, stopwords=True, **karwgs):
        super(BOW6kFeature, self).__init__(**karwgs)
        self.stopwords = stopwords
        self.feature_name = self.feature_name + '-%s' % (stopwords)

    def load_instances(self, train_instances):

        ''' extract features from train_set '''
        if self.load is False or not os.path.isfile(self.feature_file):

            ''' extract features to features '''
            features = []
            infos = []
            feature_file = self.feature_file.replace("6k", "")
            _features, _n_dim, _n_instance = Feature.load_feature_from_file(feature_file)
            print(_features[0], _n_dim)
            ''' get features from train instances'''
            for _feature in _features:
                feature = Feature._feat_string_to_list(_feature, _n_dim)
                features.append(feature[:11])
                infos.append([])

            print(len(features), features[0])
            ''' write features to file '''
            Feature.write_feature_to_file(self.feature_file, features, infos)

        ''' load features from file '''
        features, n_dim, n_instance = Feature.load_feature_from_file(self.feature_file)
        return features, n_dim, n_instance


class POSVector6kFeature(Feature):
    def load_instances(self, train_instances):

        ''' extract features from train_set '''
        if self.load is False or not os.path.isfile(self.feature_file):

            ''' extract features to features '''
            features = []
            infos = []
            feature_file = self.feature_file.replace("6k", "")
            _features, _n_dim, _n_instance = Feature.load_feature_from_file(feature_file)
            print(_features[0], _n_dim)
            ''' get features from train instances'''
            for _feature in _features:
                feature = Feature._feat_string_to_list(_feature, _n_dim)
                features.append(feature[:6])
                infos.append([])

            print(len(features), features[0])
            ''' write features to file '''
            Feature.write_feature_to_file(self.feature_file, features, infos)

        ''' load features from file '''
        features, n_dim, n_instance = Feature.load_feature_from_file(self.feature_file)
        return features, n_dim, n_instance


class DependencyRelation6kFeature(Feature):
    def __init__(self, convey='count', **karwgs):
        super(DependencyRelation6kFeature, self).__init__(**karwgs)
        self.convey = convey
        self.feature_name = self.feature_name + '-%s' % (convey)

    def load_instances(self, train_instances):

        ''' extract features from train_set '''
        if self.load is False or not os.path.isfile(self.feature_file):

            ''' extract features to features '''
            features = []
            infos = []
            feature_file = self.feature_file.replace("6k", "")
            _features, _n_dim, _n_instance = Feature.load_feature_from_file(feature_file)
            print(_features[0], _n_dim)
            ''' get features from train instances'''
            for _feature in _features:
                feature = Feature._feat_string_to_list(_feature, _n_dim)
                features.append(feature[:11])
                infos.append([])

            print(len(features), features[0])
            ''' write features to file '''
            Feature.write_feature_to_file(self.feature_file, features, infos)

        ''' load features from file '''
        features, n_dim, n_instance = Feature.load_feature_from_file(self.feature_file)
        return features, n_dim, n_instance


class DependencyGram6kFeature(Feature):
    def __init__(self, convey='count', **karwgs):
        super(DependencyGram6kFeature, self).__init__(**karwgs)
        self.convey = convey
        self.feature_name = self.feature_name + '-%s' % (convey)

    def load_instances(self, train_instances):

        ''' extract features from train_set '''
        if self.load is False or not os.path.isfile(self.feature_file):

            ''' extract features to features '''
            features = []
            infos = []
            feature_file = self.feature_file.replace("6k", "")
            _features, _n_dim, _n_instance = Feature.load_feature_from_file(feature_file)
            print(_features[0], _n_dim)
            ''' get features from train instances'''
            for _feature in _features:
                feature = Feature._feat_string_to_list(_feature, _n_dim)
                features.append(feature[:11])
                infos.append([])

            print(len(features), features[0])
            ''' write features to file '''
            Feature.write_feature_to_file(self.feature_file, features, infos)

        ''' load features from file '''
        features, n_dim, n_instance = Feature.load_feature_from_file(self.feature_file)
        return features, n_dim, n_instance

