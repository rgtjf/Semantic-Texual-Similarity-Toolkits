# coding: utf8
from __future__ import print_function

from collections import Counter

from stst.features.features import Feature
from stst import dict_utils, utils
from stst import config

class NERFeature(Feature):

    def extract_information(self, train_instances):
        dict_ner = {}
        with utils.create_read_file(config.NER_FILE) as f:
            for line in f:
                items = line.strip().split('\t')
                word = items[0]
                print(items, len(items))
                for w in items[1].split():
                    dict_ner[w] = word
        self.dict_ner = dict_ner

    def replace(self, sent):
        res = []
        for word in sent:
            cur_word = word
            if word in self.dict_ner:
                cur_word = self.dict_ner[word]
            res.append(cur_word)
        return res

    def extract(self, train_instance):
        ner_sa, ner_sb = train_instance.get_list('ner')
        word_sa, word_sb = train_instance.get_list('word')
        word_sa = self.replace(word_sa)
        word_sb = self.replace(word_sb)

        num_all_ner = 0
        num_in_ner = 0

        for word, ner in zip(word_sa, ner_sa):
            if ner != "" and word not in ['云栖', '杭州']:
                if word in word_sb: num_in_ner += 1
                num_all_ner += 1

        score = 0.0
        if num_in_ner == 0 and num_all_ner != 0:
            score = -1e8
        # score = 1. * num_in_ner / num_all_ner if num_all_ner != 0 else 0.0

        features = [score]
        infos = [num_all_ner, num_in_ner]
        return features, infos


class BiNERFeature(Feature):

    def extract(self, train_instance):
        pass