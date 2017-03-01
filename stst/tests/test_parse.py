# coding: utf8
from __future__ import print_function

from features_basic import *
from features_dependency import *
from features_embedding import *
from features_mt import *
from features_ner import *
from features_ngram import *
from features_nn import *
from features_pos import *
from features_remove_nonlinear_kernel import *
from features_sequence import *
from features_tree_kernels import *

if __name__ == '__main__':


    train = config.SNLI_TRAIN_FILE
    dev = config.SNLI_DEV_FILE
    test = config.SNLI_TEST_FILE

    dev_parse_data = data_utils.load_parse_data(dev, None, flag=False)


    train_parse_data = data_utils.load_parse_data(train, None, flag=True)
    test_parse_data = data_utils.load_parse_data(test, None, flag=True)
