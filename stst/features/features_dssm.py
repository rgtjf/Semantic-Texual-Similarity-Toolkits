# coding: utf8
from __future__ import print_function

import os
from collections import Counter

from stst.features.features import Feature
from stst import dict_utils, utils
from stst import config

class DSSMFeature(Feature):

    def extract_instances(self, train_instances):
        """
        extract features to features
        """
        features, infos = [], []
        dssm_score_file = config.DSSM_SCORE_FILE
        with utils.create_read_file(dssm_score_file) as f:
            for line in f:
                score = eval(line.split()[-1])
                features.append([score])
                infos.append(['dssm_score'])
        return features, infos


class LSTMFeature(Feature):

    def __init__(self, name, file_name, **kwargs):
        super(LSTMFeature, self).__init__(**kwargs)
        self.name = name
        self.file_name = file_name
        self.feature_name = self.feature_name + '-%s' % (name)

    def extract_instances(self, train_instances):
        """
        extract features to features
        """
        features, infos = [], []
        score_file_path = os.path.join(config.SCORE_DIR, self.file_name)
        # score_file = config.LSTM_SCORE_FILE
        with utils.create_read_file(score_file_path) as f:
            for line in f:
                score = eval(line.split()[0])
                features.append([score])
                infos.append(['lstm_score'])
        return features, infos
