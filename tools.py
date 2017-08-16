# coding: utf8
from __future__ import print_function
from __future__ import unicode_literals

import stst
import config
from stst.features.features_sequence import SequenceFeature, SentenceFeature
from stst.features.features_ngram import nGramOverlapFeature, nCharGramOverlapFeature

from stst.evaluation import Evaluation

# Define Model
gb = stst.Classifier(stst.AverageEnsemble())
model = stst.Model('U', gb)

# Add features to the Model
model.add(nGramOverlapFeature(load=False))
# model.add(nCharGramOverlapFeature(load=False))
# model.add(SentenceFeature())

# train and test
test_file = config.CCO_FILE

# parse data
test_instances = stst.load_parse_data(test_file, flag=False)

sents = []
for instance in test_instances:
    sa, sb = instance.get_pair('char')
    sents.append(sa)
    sents.append(sb)

idf_dict = stst.utils.idf_calculator(sents)
idf_list = sorted(idf_dict.items(), key=lambda x:x[1], reverse=True)
for x in idf_list:
    print(x[0], x[1])

#
# lst = []
# with open('hard_list.txt') as f:
#     for line in f:
#         lst.append('qqid:'+line.strip())
#     print(lst[0])
#
# for instance in test_instances:
#     if instance.get_score() == 1 and instance.get_qqid() in lst:
#         print(instance.get_instance_string())