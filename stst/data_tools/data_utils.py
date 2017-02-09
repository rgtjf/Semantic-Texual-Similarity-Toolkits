# coding: utf8
from __future__ import print_function

import codecs
import traceback

import json
import pyprind

from ..lib.pycorenlp.corenlp_utils import nlp
from .. import utils
from sent_pair import SentPair


def load_data(train_file):
    """
    Return list of dataset given train_file and gs_file
    Value: [(sa:str, sb:str, score:float)]
    """
    with codecs.open(train_file, 'r', encoding='utf8') as f:
        data = []
        for idx, line in enumerate(f):
            line = line.strip().split('\t')
            score = 0.
            if len(line) == 3:
                score = float(line[2])
            sa, sb = line[0], line[1]
            data.append((sa, sb, score))
    return data


def load_parse_data(train_file, flag=False):
    """
    Load data after Parse, like POS, NER, etc.
    Value: [ SentPair:class, ... ]
    Parameter:
        flag: False(Default), Load from file (resources....)
              True, Parse and Write to file, and then load from file
    """

    ''' Pre-Define Write File '''
    parse_train_file = train_file.replace('data', 'parse')

    if flag:

        print(train_file)

        ''' Parse Data '''
        data = load_data(train_file)

        print('*' * 50)
        print("Parse Data, train_file=%s, n_train=%d\n" % (train_file, len(data)))

        # idx = 0
        parse_data = []
        process_bar = pyprind.ProgPercent(len(data))
        for (sa, sb, score) in data:
            process_bar.update()
            # idx += 1
            # if idx > 20:
            #     break
            try:
                parse_sa = nlp.parse(sa)
                parse_sb = nlp.parse(sb)
            except Exception:
                print(sa, sb)
                traceback.print_exc()
                raise Exception('Check whether you have started the CoreNLP server e.g.\n'
                                '$ cd stanford-corenlp-full-2015-12-09/ \n'
                                '$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer')
            parse_data.append((parse_sa, parse_sb, score))

        ''' Write Data to File '''
        with utils.create_write_file(parse_train_file) as f_parse:
            for parse_instance in parse_data:
                line = json.dumps(parse_instance)
                print(line, file=f_parse)

    ''' Load Data from File '''
    print('*' * 50)
    parse_data = []
    with codecs.open(parse_train_file, 'r', encoding='utf8') as f:
        for line in f:
            parse_json = json.loads(line)
            sentpair_instance = SentPair(parse_json)
            parse_data.append(sentpair_instance)

    print("Load Data, train_file=%s, n_train=%d\n" % (train_file, len(parse_data)))
    return parse_data
