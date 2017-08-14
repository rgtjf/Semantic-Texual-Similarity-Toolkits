# coding: utf8
from __future__ import print_function

import csv
import numpy as np

from stst import utils, config


def record(record_file, dev_pearsonr, test_pearsonr, model):
    with utils.create_write_file(record_file, 'ab') as f:
        writer = csv.writer(f, delimiter=',')
        features = [ feature.feature_name for feature in model.feature_list ]
        writer.writerow([ model.model_name, dev_pearsonr, test_pearsonr, model.classifier.strategy.trainer, features ])
