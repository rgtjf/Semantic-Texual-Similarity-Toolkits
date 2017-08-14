# coding: utf8
from __future__ import print_function
from __future__ import unicode_literals

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import operator
import json

import matplotlib.pyplot as plt
import codecs
import argparse
import math
from collections import OrderedDict


class Evaluation(object):

    def __init__(self, score_file, gold_file):
        """

        Parameters:
           score_file: preds [score], !! the length > dev_file
                text format:
           gold_file:
                text format: LABEL qid:ID question answer
        """
        def load_scores(score_file):
            scores = []
            for line in codecs.open(score_file, encoding='utf8'):
                line = line.strip().split('\t')[0]
                scores.append(eval(line))
            return scores

        preds = load_scores(score_file)

        question_dict = OrderedDict()
        index = 0
        for line in codecs.open(gold_file, encoding='utf8'):
            # '{:.3f}\t{chitchat}\t{id}\t{sa}\t{sb}'
            # 1	yunqi	791	随行 人员 需要 报名 吗 ？	338	本人 已经 报名 ， 如 有 随行 人员 是否 需要 单独 报名 ？ 或者 直接 前往 现场 报名 签到 即 可 ？

            items = line.strip().split('\t')
            id = items[2]
            dict_pair = {'score': preds[index],
                         'sa': items[3], 'sb': items[5],
                         'id': items[4], 'type': items[1],
                         'qid': items[2],
                         'gold': int(eval(items[0])),
                         }
            if id not in question_dict:
                question_dict[id] = []
            question_dict[id].append(dict_pair)
            index += 1

        for k, v in question_dict.items():
            v.sort(key=lambda x: x['score'], reverse=True)

        self.ranking_dict = question_dict
        self.chitchat_list = []
        self.yunqi_list = []
        for k, v in question_dict.items():
            if v[0]['type'] == 'yunqi':
                self.yunqi_list.append(v)
            elif v[0]['type'] == 'chitchat':
                self.chitchat_list.append(v)
            else:
                raise NotImplementedError

    def eval_P(self, threshold=0.0):
        # recall, precision
        chitchat_unk = 0
        chitchat_all = 0

        yunqi_all = 0
        yunqi_unk = 0
        yunqi_acc = 0
        yunqi_wro = 0

        for k, v in self.ranking_dict.items():
            if v[0]['type'] == 'yunqi':
                yunqi_all += 1
                if v[0]['score'] > threshold:
                    if v[0]['gold'] == 1:
                        yunqi_acc += 1
                    else:
                        yunqi_wro += 1

                else:
                    yunqi_unk += 1

            elif v[0]['type'] == 'chitchat':
                chitchat_all += 1
                if v[0]['score'] <= threshold:
                    chitchat_unk += 1

        print('chitchat: {:3d} / {:3d}'.format(chitchat_unk, chitchat_all), end='\t\t')
        print('yunqi: {:3d}+{:3d} {:3d} / {:3d}'.format(yunqi_acc, yunqi_wro, yunqi_unk, yunqi_all))

        return None

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
        for v in self.yunqi_list:
            labels = [x['gold'] for x in v[:N]]
            if 1 in labels:
                p += 1
            else:
                pass
            s += 1
        p_at_n = 1. * p / s if s > 0 else 0
        return p_at_n

    def eval_MAP_yunqi(self):
        scores = []
        missing_MAP = 0
        for v in self.yunqi_list:
            item = [x['gold'] for x in v]
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

    def eval_MRR_yunqi(self):
        scores = []
        for v in self.yunqi_list:
            item = [x['gold'] for x in v]
            for i, val in enumerate(item):
                if val == 1:
                    scores.append(1.0 / (i + 1))
                    break
        return sum(scores) / len(scores) if len(scores) > 0 else 0.0

    def eval_yunqi(self):
        """
        when the recall is 64, calucate the precision
        :return:
        """
        top1_list = [v[0] for v in self.yunqi_list]
        num_yunqi = len(top1_list)
        sort_top1_list = sorted(top1_list, key=lambda x:x['score'], reverse=True)
        num_recall = int(math.ceil(0.8 * num_yunqi))
        sort_top1_list = sort_top1_list[:num_recall]

        num_acc = 0.
        for v in sort_top1_list:
            if v['gold'] == 1:
                num_acc += 1
        precision = num_acc / num_recall
        print('recall: {:4d}/{:4d}, precision: {:.4f}'.format(num_recall, num_yunqi, precision))
        return precision


    def case_study_chitchat(self):
        chitchat_best_list = []
        for v in self.chitchat_list:
            chitchat_best_list.append(v[0])
        chitchat_best_list.sort(key=lambda x:x['score'], reverse=True)

        for top_1 in chitchat_best_list:
            print('{}\t{}\n'
                  'pred: {}\t{}\t{}'.format(top_1['type'], top_1['id'],
                                            top_1['sa'], top_1['sb'], top_1['score']))

    def case_study_yunqi(self):
        """
        return the wrong questions
        qqid gold_q(*), user_query, pred_q, pos_gold_q
        """
        for v in self.yunqi_list:
            top_1 = v[0]
            if top_1['gold'] == 1:
                continue

            top_g = None
            for x in v:
                if x['gold'] == 1:
                    top_g = x
                    break

            print('{}\tgold: {}\n'
                  'pred: {}\t{}\t{}\n'
                  'gold: {}\t{}\t{}'.format(top_g['type'], top_g['id'],
                                            top_1['sa'], top_1['sb'], top_1['score'],
                                            top_g['sa'], top_g['sb'], top_g['score']))



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--score_file', help='PATH to score_file', type=str, required=True)
    args.add_argument('--test_file', help='PATH to test_file', type=str, required=True)

    FLAGS, _ = args.parse_known_args()

    e = Evaluation(FLAGS.score_file, FLAGS.test_file)

    e.case_study_chitchat()

    e.case_study_yunqi()

    for th in range(0, 101, 10):
        e.eval_P(th / 100.)

    e.eval_yunqi()

