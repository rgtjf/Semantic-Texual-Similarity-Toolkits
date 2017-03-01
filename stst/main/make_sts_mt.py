# coding: utf8
from __future__ import print_function

from classifier import *
from features_mt import *
from model import Model
from data_utils import *
from main_tools import *

if __name__ == '__main__':
    sklearn_gb = Classifier(sklearn_GB())
    model = Model('MT-gb', sklearn_gb)
    ''' MT Feature '''
    model.add(AsiyaMTFeature())

    # train_file = config.STS_TRAIN_FILE
    # dev_file = config.STS_DEV_FILE
    # test_file = config.STS_TEST_FILE
    # train_instances = load_parse_data(train_file)
    # dev_instances = load_parse_data(dev_file)
    # test_instances = load_parse_data(test_file)

    train_sts(model)
    test_sts(model)
    predict_sts(model)
