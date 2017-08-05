# coding: utf8
from __future__ import print_function

import codecs
import json
import os

import pyprind

from stst import utils
from stst.data_tools.sent_pair import SentPair


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
    parse_train_file = train_file.replace('./data', './generate/parse')

    if flag or not os.path.isfile(parse_train_file):

        ''' Parse Data '''
        parse_data = []
        data = load_data(train_file)

        print('*' * 50)
        print("Parse Data, train_file={}, n_train={:d}".format(train_file, len(data)))

        process_bar = pyprind.ProgPercent(len(data))
        for (sa, sb, score) in data:
            process_bar.update()
            dict_pair = {'sa': sa, 'sb': sb, 'score': score}
            parse_data.append(dict_pair)

        ''' Write Data to File '''
        with utils.create_write_file(parse_train_file) as f_parse:
            for parse_instance in parse_data:
                line = json.dumps(parse_instance)
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