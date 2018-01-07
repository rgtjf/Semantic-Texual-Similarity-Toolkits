# coding: utf8
from __future__ import print_function

import json
import os

import pyprind

from stst import utils, config


class Feature(object):
    def __init__(self,
                 load=True,
                 dimension=None,
                 **kwargs):
        self.load = load
        self.feature_name = self.__class__.__name__
        self.dimension = dimension
        self.kwargs = kwargs

    def extract_dataset_instances(self, train_instances, train_file, is_training=False):
        """
        ../resources/data/sts-en-en/.. -> ../features/sts-en-en/..
        """
        # Need This to help global function
        self.train_file = train_file
        # Re-define self.feature_file to prevent self.feature_file change two times
        # (i.e., train and test)
        train_file_name = os.path.basename(train_file)
        train_file_name = os.path.splitext(train_file_name)[0]
        self.feature_file = '{}/{}/{}.txt'.format(
            config.FEATURE_DIR, train_file_name, self.feature_name)

        return self.load_instances(train_instances)

    def load_instances(self, train_instances):
        """
        extract features from train_set
        """
        if self.load is False or not os.path.isfile(self.feature_file):
            print(self.feature_file)

            ''' extract features to features '''
            features, infos = self.extract_instances(train_instances)

            ''' write features to file '''
            Feature.write_feature_to_file(self.feature_file, features, infos)

        ''' load features from file '''
        features, n_dim, n_instance = Feature.load_feature_from_file(self.feature_file)
        return features, n_dim, n_instance

    def extract_instances(self, train_instances):
        """
        extract features to features
        """
        # first extract information from train_instance
        # for only be used to extract data_set information and can reuse the pyprind
        self.extract_information(train_instances)
        features = []
        infos = []
        process_bar = pyprind.ProgPercent(len(train_instances))
        for train_instance in train_instances:
            process_bar.update()
            feature, info = self.extract(train_instance)  # 可变参数进行传递！
            features.append(feature)
            infos.append(info)
        return features, infos

    def extract_information(self, train_instances):
        """
        extract information from train_instances
        """
        pass

    def extract(self, train_instance):
        """
        extract feature from a instance
        """
        pass

    @staticmethod
    def write_feature_to_file(feature_file, features, infos):
        """
        write features string to file
        """
        if type(features[0]) is list:
            dim = len(features[0])
        else:
            dim = infos[0][0]

        f_feature = utils.create_write_file(feature_file)

        ''' write features infomation to file '''
        print(len(features), dim, file=f_feature)

        ''' write features string to file '''
        for feature, info in zip(features, infos):
            ''' type(feature) is list '''
            if type(feature) is list:
                feature_string = Feature._feat_list_to_string(feature)
            elif type(feature) is str:
                feature_string = feature
            else:
                raise NotImplementedError

            info_string = Feature._info_list_to_string(info)
            print(feature_string + '\t#\t' + info_string, file=f_feature)

        f_feature.close()

    @staticmethod
    def load_feature_from_file(feature_file):
        """
        load features from file
        """
        f_feature = utils.create_read_file(feature_file)

        feature_information = f_feature.readline()
        n_instance, n_dim = feature_information.strip().split()
        n_instance, n_dim = int(n_instance), int(n_dim)

        features = []
        for feature in f_feature:
            feature_string, instance_string = feature.split("\t#\t")
            features.append(feature_string)

        return features, n_dim, n_instance

    @staticmethod
    def _feat_list_to_string(feat_list):
        """
         [0, 1, 0, 1] => 2:1 4:1
        """
        feat_dict = {}

        for index, item in enumerate(feat_list):
            if item != 0:
                feat_dict[index + 1] = item

        transformed_list = [str(key) + ":" + str(feat_dict[key]) for key in sorted(feat_dict.keys())]
        feat_string = " ".join(transformed_list)

        return feat_string

    @staticmethod
    def _feat_dict_to_string(feat_dict):
        transformed_list = ['{}:{}'.format(key+1, feat_dict[key]) for key in sorted(feat_dict.keys())]
        feat_string = ' '.join(transformed_list)
        return feat_string

    @staticmethod
    def _feat_string_to_list(feat_string, ndim):
        feat_list = [0] * ndim
        for feat in feat_string.split():
            index, value = feat.split(':')
            index = int(index) - 1
            value = eval(value)
            feat_list[index] = value
        return feat_list

    @staticmethod
    def _info_list_to_string(info_list):
        info_string = json.dumps(info_list, ensure_ascii=False)
        return info_string


class CustomFeature(Feature):
    def extract(self, train_instance):
        """
        Extract features and info from one train instance
        """
        features = 'features'
        infos = 'infos'

        return features, infos