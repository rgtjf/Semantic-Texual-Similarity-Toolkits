# coding: utf8
from __future__ import print_function

import sys
sys.path.append('..')

import codecs
import scipy.stats as meas

import stst

class DefineExample(stst.Example):

    def __init__(self, sa, sb, score):
        """
        sa, sb: word list
        score: float
        """
        self.sa = sa
        self.sb = sb
        self.score = score

    def get_words(self):
        """ Return sa, sb """
        return self.sa, self.sb

    def get_score(self):
        """ Return the gold score """
        return self.score

    def get_instance_string(self):
        """ Return instance string """
        instance_string = "{}\t{}\t{}".format(self.score, self.sa, self.sb)
        return instance_string

    @staticmethod
    def load_data(file_path):
        """ Return list of examples """
        examples = []
        with codecs.open(file_path, encoding='utf8') as f:
            for line in f:
                items = line.strip().split('\t')
                sa, sb, score = items[5], items[6], float(items[4])
                example = DefineExample(sa, sb, score)
                examples.append(example)
        return examples               


class STSEvaluation(stst.Evaluation):

    def __init__(self, predicts, golds):
        self.predicts = predicts
        self.golds = golds
        # predicts = open(score_file).readlines()
        # golds = open(gold_file).readlines()
        # self.predicts = [float(x.strip().split()[0])for x in predicts]
        # self.golds = [float(x.strip().split()[0]) for x in golds]
    
    def evaluation(self):
        result = {}
        result['pearsonr'] = self.pearsonr(self.predicts, self.golds)
        return result
    
    @staticmethod
    def pearsonr(predict, gold):
        """
        pearsonr of predict and gold
        :param predict: list
        :param gold: list
        :return: mape
        """
        pearsonr = meas.pearsonr(predict, gold)[0]
        return pearsonr
    
    @staticmethod
    def init_from_modelfile(output_file):
        """
        output_file: 
            FORMAT 
                25.00   #   1.8 A woman is cutting onions.  A woman is cutting tofu.
        """
        predicts, golds = [], []
        with open(output_file) as f:
            for line in f:
                items = line.strip().split('\t#\t')
                predict = float(items[0])
                gold = float(items[1].split()[0])
                predicts.append(predict)
                golds.append(gold)
        return STSEvaluation(predicts, golds)


class DefineFeature(stst.Feature):
    def extract(self, train_instance):
        sa, sb = train_instance.get_words()
        features = [len(sa), len(sb)]
        infos = ['test']
        return features, infos



classifier = stst.Classifier(stst.classifier.AverageEnsemble())

model = stst.Model('basic_model', classifier)

train_file = '../data/stsbenchmark/sts-test.csv'
train_instances = DefineExample.load_data(train_file)

model.add(DefineFeature(load=False))

model.test(train_instances, train_file)

evaluation = STSEvaluation.init_from_modelfile(model.output_file)
results = evaluation.evaluation()
print(results)


logger = stst.utils.get_logger('stst-basic.log')

logger.info('===<start>===')
logger.info('model_name = %s' % model.model_name)
logger.info('classfier_name = %s' % model.classifier.strategy.trainer)
logger.info('feature_list = %s' % [feature.feature_name for feature in model.feature_list])
logger.info('results = %s' % results)
logger.info('===<end>===\n\n')
