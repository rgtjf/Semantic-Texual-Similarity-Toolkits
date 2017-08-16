# coding: utf8
from __future__ import print_function
from __future__ import unicode_literals

import scipy.stats as meas
import operator
import json
import matplotlib.pyplot as plt
import codecs

def eval_pearsonr(predict, gold):
    """
    pearsonr of predict and gold
    :param predict: list
    :param gold: list
    :return: mape
    """
    pearsonr = meas.pearsonr(predict, gold)[0]
    return pearsonr


def eval_file(predict_file, gold_feature_file):
    predict = open(predict_file).readlines()
    gold = open(gold_feature_file).readlines()
    predict = [float(x.strip().split()[0])for x in predict]
    gold = [float(x.strip().split()[0]) for x in gold]
    pearsonr = eval_pearsonr(predict, gold)
    return pearsonr


def eval_output_file(predict_file):
    predict, gold = [], []
    with open(predict_file) as f:
        for line in f:
            line = line.strip().split('\t#\t')
            predict.append(float(line[0]))
            gold.append(float(line[1].split('\t')[0]))
    pearsonr = eval_pearsonr(predict, gold)
    return pearsonr


def eval_file_corpus(predict_file_list, gold_file_list):
    predicts, golds = [], []
    for predict_file, gold_file in zip(predict_file_list, gold_file_list):
        predict = open(predict_file).readlines()
        gold = open(gold_file).readlines()
        predicts += predict
        golds += gold
    predicts = [float(x.strip().split()[0]) for x in predicts]
    golds = [float(x.strip().split()[0]) for x in golds]
    pearsonr = eval_pearsonr(predicts, golds)
    return pearsonr


class Evaluation(object):

    def __init__(self, output_file, gold_file=None):
        """

        Parameters:
           output_file: preds [score], !! the length > dev_file
                text format:
           gold_file:
                text format: LABEL qid:ID question answer
        """
        def load_scores(score_file):
            scores = []
            for line in codecs.open(score_file, encoding='utf8'):
                line = line.strip().split('\t#\t')[0]
                scores.append(eval(line))
            return scores

        def load_scores_dssm(score_file):
            scores = []
            for line in open(score_file):
                score_list = json.loads(line.strip())
                for score in score_list:
                    scores.append(score)
            return scores

        preds = load_scores(output_file)

        # gold_file: load output_file from stst
        question_dict = {}
        index = 0
        for line in codecs.open(gold_file, encoding='utf8'):
            items = line.strip().split('\t#\t')
            gold, qid, sa, sb = items[1].split('\t')
            if qid not in question_dict:
                question_dict[qid] = []
            gold = int(eval(gold))
            question_dict[qid].append((preds[index], gold))
            index += 1

        for k, v in question_dict.items():
            v.sort(key=operator.itemgetter(0), reverse=True)

        self.ranking_dict = question_dict

    def eval_P(self, threshold=0.0):
        # recall, precision
        num_true_labels = 0
        num_all_labels  = 0
        num_miss_labels = 0

        for k, v in self.ranking_dict.items():
            top_1 = v[0]
            if top_1[0] >= threshold:
                if top_1[1] == 1:
                    num_true_labels += 1
            else:
                num_miss_labels += 1
            num_all_labels += 1

        num_calc_lables = num_all_labels - num_miss_labels
        precision = 1. * num_true_labels / num_calc_lables if num_calc_lables > 0 else 0
        recall = num_calc_lables / num_all_labels

        print(threshold, precision, recall)
        return precision, recall

    def plot(self, step=1):
        x = []
        y1 = []
        y2 = []
        for i in range(0, 101, step):
            precision, recall = self.eval_P(1.*i/100)
            x.append(1.*i/100)
            y1.append(precision)
            y2.append(recall)

        colors = ['red', 'blue', 'black', 'navy', 'turquoise', 'darkorange']
        lw = 1
        plt.clf()

        plt.plot(x, y1, color=colors[0], lw=lw, label='precision')
        plt.plot(x, y2, color=colors[1], lw=lw, label='recall')

        plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('Treshold')
        plt.ylabel('Precision')
        plt.title('Precision (Recall) - Treshold Curve')
        plt.legend(loc="lower right")
        # plt.savefig('eval_plot')
        plt.show()

    def eval_P_at_N(self, N=1):
        """

        :param ranking_data: {qid: [(pred, gold)]}
        :param N:
        :return:
        """
        p = 0.
        s = 0.
        for k, v in self.ranking_dict.items():
            labels = [x[1] for x in v[:N]]
            if 1 in labels:
                p += 1
            else:
                pass
            s += 1
        p_at_n = 1. * p / s if s > 0 else 0
        return p_at_n

    def eval_MAP(self):
        scores = []
        missing_MAP = 0
        for k, v in self.ranking_dict.items():
            item = [x[1] for x in v]
            count = 0.0
            temp = []
            for i, val in enumerate(item):
                if val == 1:
                    count += 1.0
                    temp.append(count / (i + 1))
            if len(temp) > 0:
                scores.append(sum(temp) / len(temp))
            else:
                missing_MAP += 1
        return sum(scores) / len(scores) if len(scores) > 0 else 0.0

    def eval_MRR(self):
        scores = []
        for k, v in self.ranking_dict.items():
            item = [x[1] for x in v]
            for i, val in enumerate(item):
                if val == 1:
                    scores.append(1.0 / (i + 1))
                    break
        return sum(scores) / len(scores) if len(scores) > 0 else 0.0


class AdvancedEvaluation(object):

    def __init__(self, output_file):
        question_dict = {}
        for line in codecs.open(output_file, encoding='utf8'):
            items = line.strip().split('\t#\t')
            dict_pair = {}
            dict_pair['score'] = items[0]
            # '{:.3f}\t{}\t{}\t{}'
            gold, qqid, sa, sb = items[1].split('\t')
            dict_pair = {'score': eval(items[0]), 'sa': sa, 'sb': sb, 'qqid': qqid, 'gold': int(eval(gold))}
            if qqid not in question_dict:
                question_dict[qqid] = []
            question_dict[qqid].append(dict_pair)

        for k, v in question_dict.items():
            v.sort(key=lambda x: x['score'], reverse=True)

        self.ranking_dict = question_dict

    def case_study(self):
        """
        return the wrong questions
        qqid gold_q(*), user_query, pred_q, pos_gold_q
        """
        for k, v in self.ranking_dict.items():
            top_1 = v[0]
            if top_1['gold'] == 1:
                continue

            top_g = None
            for x in v:
                if x['gold'] == 1:
                    top_g = x
                    break

            print('{}\n'
                  'pred: {}\t{}\t{}\n'
                  'gold: {}\t{}\t{}'.format(top_1['qqid'],
                                            top_1['sa'], top_1['sb'], top_1['score'],
                                            top_g['sa'], top_g['sb'], top_g['score']))


class AdvancedDSSMEvaluation(object):

    def __init__(self, dssm_file, output_file):
        def load_scores(score_file):
            scores = []
            for line in codecs.open(score_file, encoding='utf8'):
                line = line.strip().split('\t#\t')[0]
                scores.append(eval(line))
            return scores

        def load_scores_dssm(score_file):
            scores = []
            for line in open(score_file):
                score_list = json.loads(line.strip())
                for score in score_list:
                    scores.append(score)
            return scores

        preds = load_scores(output_file)

        question_dict = {}
        index = 0
        for line in codecs.open(output_file, encoding='utf8'):
            items = line.strip().split('\t#\t')
            dict_pair = {}
            dict_pair['score'] = items[0]
            # '{:.3f}\t{}\t{}\t{}'
            gold, qqid, sa, sb = items[1].split('\t')
            dict_pair = {'score': preds[index], 'sa': sa, 'sb': sb, 'qqid': qqid, 'gold': int(eval(gold))}
            if qqid not in question_dict:
                question_dict[qqid] = []
            question_dict[qqid].append(dict_pair)
            index += 1

        for k, v in question_dict.items():
            v.sort(key=lambda x: x['score'], reverse=True)

        self.ranking_dict = question_dict

    def case_study(self):
        """
        return the wrong questions
        qqid gold_q(*), user_query, pred_q, pos_gold_q
        """
        for k, v in self.ranking_dict.items():
            top_1 = v[0]
            if top_1['gold'] == 1:
                continue

            top_g = None
            for x in v:
                if x['gold'] == 1:
                    top_g = x
                    break

            print('{}\n'
                  'pred: {}\t{}\t{}\n'
                  'gold: {}\t{}\t{}'.format(top_1['qqid'],
                                            top_1['sa'], top_1['sb'], top_1['score'],
                                            top_g['sa'], top_g['sb'], top_g['score']))