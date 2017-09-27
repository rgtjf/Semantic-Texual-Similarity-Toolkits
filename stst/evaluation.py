# coding: utf8
from __future__ import print_function




class Evaluation(object):

    def __init__(self, score_file, gold_file):
        
        self.predicts = None
        self.golds = None
  
    def evaluation(self):
        return {}

    @staticmethod
    def create_from_predicts(predicts, score_file, gold_file):
        """ 
        params:
            predicts: nparray or list
            score_file: output_file, str
            gold_file: gold standard file, str
        """
        golds = []
        N = len(open(gold_file).readlines())

        predicts = predicts[:N]
        print('The number of predicts has exceeded that of golds', predicts, N)
        
        with open(score_file) as fw:
            for predict in predicts:
                print(predict, file=fw)

        return Evaluation(score_file, gold_file)
