# coding: utf8
from __future__ import print_function

import codecs
import json
import os
from collections import OrderedDict

import pyprind

from stst import utils
from stst.data_tools.sent_pair import SentPair

def load_corpus(corpus_file):
    """
    {"text": "你可以做哪些事啊", "id": "245", "parsed": [{"index": 1, "head": 2, "word": "你", "pos": "PN", "label": "SUB", "ner": ""}, {"index": 2, "head": 0, "word": "可以", "pos": "VV", "label": "ROOT", "ner": ""}, {"index": 3, "head": 2, "word": "做", "pos": "VV", "label": "VC", "ner": ""}, {"index": 4, "head": 5, "word": "哪些", "pos": "DT", "label": "NMOD", "ner": ""}, {"index": 5, "head": 3, "word": "事", "pos": "NN", "label": "OBJ", "ner": ""}, {"index": 6, "head": 2, "word": "啊", "pos": "SP", "label": "DEP", "ner": ""}]}
    """
    corpus = OrderedDict()
    with codecs.open(corpus_file, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            corpus[item['id']] = item
    return corpus


def load_data(train_file, corpus_file):
    """

    :param train_file:
    :param corpus_file:
    :return:
    """

    corpus = load_corpus(corpus_file)

    yunqi_standard = []
    with codecs.open(train_file, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f):
            item = line.strip().split(chr(3))
            if item[0] == 'yunqi':
                yunqi_standard.append(item[1])

    data = []
    with codecs.open(train_file, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f):
            item = line.strip().split(chr(3))
            type = item[0]
            standard_id = item[1]

            # sa is the user question
            # sb is the standard question
            for sa_id in item[2].split(chr(1)):
                sa = corpus[sa_id]

                # iterative all the YunQi standard questions
                for candinate_id in yunqi_standard:
                    label = 0
                    if type == 'yunqi' and standard_id == candinate_id:
                        label = 1
                    sb = corpus[candinate_id]
                    obj = {'type':type, 'label': label, 'sa':sa, 'sb':sb, 'sa_id': sa_id, 'sb_id':candinate_id}
                    data.append(obj)
    return data


def load_parse_data(train_file, corpus_file, flag=False):
    """
    Load data after Parse, like POS, NER, etc.
    Value: [ SentPair:class, ... ]
    Parameter:
        flag: False(Default), Load from file (resources....)
              True, Parse and Write to file, and then load from file
    """

    ''' Pre-Define Write File '''
    parse_train_file = train_file.replace('./data', './generate/parse')

    if flag or not os.path.isfile(parse_train_file):

        ''' Parse Data '''
        parse_data = load_data(train_file, corpus_file)

        print('*' * 50)
        print("Parse Data, train_file={}, n_train={:d}".format(train_file, len(parse_data)))

        ''' Write Data to File '''
        with utils.create_write_file(parse_train_file) as f_parse:
            for parse_instance in parse_data:
                line = json.dumps(parse_instance)
                print(line, file=f_parse)

        ''' Write Data to File '''
        simple_train_file = parse_train_file.replace('ids', 'txt')
        print("simple_train_file={}, n_train={:d}".format(simple_train_file, len(parse_data)))
        with utils.create_write_file(simple_train_file) as f_parse:
            for parse_instance in parse_data:
                sa = [item['word'] for item in parse_instance['sa']['parsed']]
                sb = [item['word'] for item in parse_instance['sb']['parsed']]
                line = '{}\t{}\t{}\t{}\t{}\t{}'.format(parse_instance['label'],
                                                       parse_instance['type'],
                                                       parse_instance['sa_id'], ' '.join(sa),
                                                       parse_instance['sb_id'], ' '.join(sb))
                print(line, file=f_parse)

    ''' Load Data from File '''
    print('*' * 50)
    parse_data = []
    with utils.create_read_file(parse_train_file) as f:
        for line in f:
            parse_json = json.loads(line)
            sentpair = SentPair(parse_json)
            parse_data.append(sentpair)

    print("Load Data, train_file={}, n_train={:d}\n".format(train_file, len(parse_data)))
    return parse_data