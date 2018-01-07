# coding: utf8
from __future__ import print_function

import stst
# from models.basic_model import DefineExample
from models.basic_model import DefineFeature, OverlapFeature, MinAvgMaxEmbeddingFeature
from models.basic_model import DefineEvaluation
from models.cl_basic_model import CLMinAvgMaxEmbeddingFeature, CLCharNgramFeature
from models.cl_basic_model import DefineExample, MTMinAvgMaxEmbeddingFeature
import tensorflow as tf

FLAGS = tf.flags.FLAGS

# Parameters
tf.flags.DEFINE_string('task', 'sts-es-en', 'task')
FLAGS._parse_flags()

classifier = stst.Classifier(stst.classifier.AverageEnsemble())

model = stst.Model('basic_model', classifier)

task = FLAGS.task
train_file = '../data_B/%s/train.json' % task
dev_file = '../data_B/%s/dev.json' % task
test_file = '../data_B/%s/test.json' % task

train_file = test_file

train_instances = DefineExample.load_data(train_file)
print(len(train_instances))
# model.add(OverlapFeature(load=False))
# model.add(CLCharNgramFeature(load=False))

if task == 'sts-es-es':
    emb_300_file = '/home/junfeng/FastText/wiki.es.vec'
    emb_100_file = '/data/wiki/models/20170724/wiki.es.100.vec'
    model.add(MinAvgMaxEmbeddingFeature('es_emb_300', 300, emb_300_file, load=False))
    # model.add(MinAvgMaxEmbeddingFeature('es_emb_100', 100, emb_100_file)
elif task == 'sts-ar-ar':
    emb_300_file = '/home/junfeng/FastText/wiki.ar.vec'
    emb_100_file = '/data/wiki/models/20170724/wiki.ar.100.vec'
    # model.add(MinAvgMaxEmbeddingFeature('ar_emb_300', 300, emb_300_file, load=False))
    # model.add(MinAvgMaxEmbeddingFeature('ar_emb_100', 100, emb_100_file)
elif task == 'sts-en-en':
    emb_300_file = '/home/junfeng/FastText/wiki.en.vec'
    model.add(MinAvgMaxEmbeddingFeature('en_emb_300', 300, emb_300_file))

elif task == 'sts-ar-en':
    ar_300_file = '/home/junfeng/FastText/wiki.ar.vec'
    en_300_file = '/home/junfeng/FastText/wiki.en.vec'
    trans_file = '../data/tm/ar.txt'
    # model.add(CLMinAvgMaxEmbeddingFeature('ar_emb_300', 300, ar_300_file, en_300_file, load=False))
    model.add(MTMinAvgMaxEmbeddingFeature('mt_ar_emb_300', 300, ar_300_file, en_300_file, load=False))

elif task == 'sts-es-en':
    es_300_file = '/home/junfeng/FastText/wiki.es.vec'
    en_300_file = '/home/junfeng/FastText/wiki.en.vec'
    trans_file = '../data/tm/es.txt'
    # model.add(CLMinAvgMaxEmbeddingFeature('es_emb_300', 300, es_300_file, en_300_file, load=False))
    model.add(MTMinAvgMaxEmbeddingFeature('mt_es_emb_300', 300, es_300_file, en_300_file, load=False))

model.test(train_instances, train_file)

evaluation = DefineEvaluation.init_from_modelfile(model.output_file)
results = evaluation.evaluation()
print(results)


logger = stst.utils.get_logger('stst-cl-basic.log')

logger.info('===<start>===')
logger.info('task = %s' % task)
logger.info('model_name = %s' % model.model_name)
logger.info('classfier_name = %s' % model.classifier.strategy.trainer)
logger.info('feature_list = %s' % [feature.feature_name for feature in model.feature_list])
logger.info('results = %s' % results)
logger.info('===<end>===\n\n')
