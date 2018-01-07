"""
how to write this file
1. define the example, like what you want to get from one example
2. init the data from different place, and return the train/dev/test set.abs
3. `write_to_json` and `read_from_json`. `read_from_json` return the example list of required data

- in the `main.py`
    only need to read_from_json
- in the 'definefeature.py'
    consider the form of the examples
"""
# coding: utf8

from __future__ import print_function

import argparse
import codecs
import math
import json

import sys
sys.path.append('..')

import stst
from stst import utils


class DefineExample(stst.Example):

    def __init__(self, example_dict):
        """
        sa, sb: word list
        score: float
        """
        self.sa = example_dict['sa']
        self.sb = example_dict['sb']
        self.en_sa = example_dict['en_sa']
        self.en_sb = example_dict['en_sb']
        self.score = example_dict['score']

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
        with utils.create_read_file(file_path, encoding='utf8') as f:
            for line in f:
                items = line.strip().split('\t')
                sa, sb, score = items[5], items[6], float(items[4])
                example_dict = {
                    'sa': sa,
                    'sb': sb,
                    'score': score
                }
                example = DefineExample(example_dict)
                examples.append(example)
        return examples               


def init_arabic_data(config_file, task):
    """
    write to train_file/dev_file/test_file
    return train_data/dev_data/test_data
    
    here I wanna to split each domain to 8:1:1
    """
    
    def load_lemma_data(file_name):
        examples = []
        with codecs.open(file_name, encoding='utf8') as f:
            for line in f:
                items = line.strip().split('\t')
                example_dict = {
                    'sa': items[0],
                    'sb': items[1],
                    'score': float(items[2])
                }
                examples.append(example_dict)
        return examples

    def load_translate_data(file_name):
        examples = []
        with codecs.open(file_name, encoding='utf8') as f:
            for line in f:
                items = line.strip().split('\t')
                example_dict = {
                    'en_sa': items[0],
                    'en_sb': items[1]
                }
                examples.append(example_dict)
        return examples

    def split_dataset(examples):
        n = len(examples)
        a = n * 8 // 10
        b = n * 9 // 10
        print(a, b, n)
        return examples[:a], examples[a:b], examples[b:]
    
    LEMMA_DIR = '../../data/lemma/sts-en-en'
    lemma_dataset = [
        'train.tsv',
        'dev.tsv',
        'test.tsv'
    ]

    TRANSLATOR_DIR = '../../data/lemma/sts-en-en'
    translate_dataset = [
        'train.tsv',
        'dev.tsv',
        'test.tsv'
    ]

    train_data = []
    dev_data = []
    test_data = []

    index = 0
    for lemma_file, translate_file in zip(lemma_dataset, translate_dataset):
        lemma_file = LEMMA_DIR + '/' + lemma_file
        translate_file = TRANSLATOR_DIR + '/' + translate_file

        lemma_examples = load_lemma_data(lemma_file)
        translate_examples = load_translate_data(translate_file)

        # `update` update the key_value, but not return the dict
        examples = [dict(lemma_example, **(translate_example)) for lemma_example, translate_example in zip(lemma_examples, translate_examples)]

        # train, dev, test = split_dataset(examples)
        index += 1
        if index == 1:
            train_data += examples
        elif index == 2:
            dev_data += examples
        else:
            test_data += examples

    return train_data, dev_data, test_data


def write_to_json(examples, output_file):
    with codecs.open(output_file, 'w', encoding='utf8') as fw:
        for example in examples:
            example_str = json.dumps(example, ensure_ascii=False)
            print(example_str, file=fw)


def read_from_json(input_file):
    examples = []
    with codecs.open(input_file, 'r', encoding='utf8') as f:
        for line in f:
            example_str = line.strip()
            example = json.loads(example_str)
            examples.append(example)
    return examples

def load_arabic_data(config_file):
   pass
    

if __name__ == '__main__':

    
    DATA_CLEAN_DIR = '../../data_clean/sts-en-en'
    train_file = DATA_CLEAN_DIR + '/train.json'
    dev_file = DATA_CLEAN_DIR + '/dev.json'
    test_file = DATA_CLEAN_DIR + '/test.json'

    train_data, dev_data, test_data = init_arabic_data('', '')
    
    write_to_json(train_data, train_file)
    write_to_json(dev_data, dev_file)
    write_to_json(test_data, test_file)
    
    train_instances = read_from_json(train_file)
    print(len(train_instances))
    print(train_instances[0])