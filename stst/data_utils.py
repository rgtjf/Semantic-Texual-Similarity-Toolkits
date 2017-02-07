# coding: utf8
from __future__ import print_function

import codecs
import re

import traceback

import utils
from lib.pycorenlp.coreNlpUtil import *
import pyprind, json
from sentpair import SentPair
import cparams

def load_data(train_file, gs_file=None):
    """
    Return list of dataset given train_file and gs_file
    Value: [(sa:str, sb:str, score:float)]
    """

    scores = None
    if gs_file is not None:
        scores = [float(x) for x in open(gs_file)]

    with codecs.open(train_file, 'r', encoding='utf8') as f:
        data = []
        for idx, line in enumerate(f):
            score = 0. if scores is None else scores[idx]
            sa, sb = preprocess(line, score)
            data.append((sa, sb, score))

    return data



def load_SICK(train_file):
    f = codecs.open(train_file, 'r', encoding='utf8').readlines()
    data = []
    fw = open(train_file.replace('/manual', ''), 'w')
    for idx, line in enumerate(f[1:]):
        pID, sa, sb, score, label = line.strip().split('\t')
        data.append((sa, sb, score))
        print(score, file=fw)
    return data


def preprocess(l, score, params=None):
    """
    Used in function load_data, to preprocess line and score
    Value: (sa, sb)
    """
    # sent_prototype = {}
    # sent_prototype['origin'] = l
    # if params['punctuations']:
    #     pass
    # if params['stopwords']:
    #     pass

    r1 = re.compile(r'\<([^ ]+)\>')
    r2 = re.compile(r'\$US(\d)')
    l = l.replace(u'’', "'")
    l = l.replace(u'``', '"')
    l = l.replace(u"''", '"')
    l = l.replace(u"´", "'")
    l = l.replace(u"—", ' ')
    l = l.replace(u"–", ' ')
    l = l.replace(u"-", " ")
    l = l.replace(u"/", " ")
    l = r1.sub(r'\1', l)
    l = r2.sub(r'$\1', l)

    #TODO next remove this
    #TODO modify stopwords and negation terms file
    # l = l.replace("n't", "not")
    # l = l.replace("'m", "am")

    sent1, sent2 = l.strip().split('\t')[0:2]

    # sent1 = sent1.split()
    # sent2 = sent2.split()
    #
    # sent1, sent2 = fix_compounds(sent1, sent2), fix_compounds(sent2, sent1)
    # sent1, sent2 = fix_compounds(sent1, sent2), fix_compounds(sent2, sent1)
    #
    # sent1 = ' '.join(sent1)
    # sent2 = ' '.join(sent2)

    return (sent1, sent2)



s = u'— adfs as\tadfa'

print(preprocess(s, 0.5))


# def fix_compounds(a, b):
#     sent2 = set(x.lower() for x in b)
#
#     a_fix = []
#     la = len(a)
#     i = 0
#     while i < la:
#         if i + 1 < la:
#             comb = a[i] + a[i + 1]
#             if comb.lower() in sent2:
#                 a_fix.append(a[i] + a[i + 1])
#                 i += 2
#                 continue
#         a_fix.append(a[i])
#         i += 1
#     return a_fix



def load_parse_data(train_file, gs_file=None, flag=False, sick=False):
    """
    Load data after Parse, like POS, NER, etc.
    Value: [ SentPair:class, ... ]
    Parameter:
        flag: False(Default), Load from file (resources....)
              True, Parse and Write to file, and then load from file
    """

    ''' Pre-Define Write File '''
    parse_train_file = train_file.replace('data', 'parse')
    parse_word_file = train_file.replace('data', 'word')
    parse_lemma_file = train_file.replace('data', 'lemma')
    parse_pos_file = train_file.replace('data', 'pos')
    parse_ner_file = train_file.replace('data', 'ner')
    parse_stopwords_lemma_file = train_file.replace('data', 'stopwords/lemma')

    if flag:

        print(train_file)
        print(gs_file)

        ''' Parse Data '''
        if 'sick' in train_file or sick:
            data = load_SICK(train_file)
        else:
            data = load_data(train_file, gs_file)

        print('*' * 50)
        print("Parse Data, train_file=%s, gs_file=%s, n_train=%d\n" % (train_file, gs_file, len(data)))

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
                #parse_sa = sa
                #parse_sb = sb
            parse_data.append((parse_sa, parse_sb, score))

        ''' Write Data to File '''

        f_parse = utils.create_write_file(parse_train_file)
        f_word = utils.create_write_file(parse_word_file)
        f_lemma = utils.create_write_file(parse_lemma_file)
        f_pos = utils.create_write_file(parse_pos_file)
        f_ner = utils.create_write_file(parse_ner_file)
        f_stopwords_lemma = utils.create_write_file(parse_stopwords_lemma_file)

        for parse_instance in parse_data:
            line = json.dumps(parse_instance)

            sentpair_instance = SentPair(parse_instance)

            score = str(sentpair_instance.get_score())
            sa_word, sb_word = sentpair_instance.get_word(type='word')
            sa_lemma, sb_lemma = sentpair_instance.get_word(type='lemma')
            sa_pos, sb_pos = sentpair_instance.get_word(type='pos')
            sa_ner, sb_ner = sentpair_instance.get_word(type='ner')

            sa_stopwords_lemma, sb_stopwords_lemma = sentpair_instance.get_word(type='lemma', stopwords=True)

            s_word = score \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sa_word, sa_word)])  \
                     + '\t'                                                      \
                     + ' '.join([w + '/' + p for w, p in zip(sb_word, sb_word)])

            s_lemma = score \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sa_word, sa_lemma)]) \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sb_word, sb_lemma)])

            s_pos = score \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sa_word, sa_pos)]) \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sb_word, sb_pos)])

            s_ner = score \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sa_word, sa_ner)]) \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sb_word, sb_ner)])

            s_stopwords_lemma = score \
                     + '\t' \
                     + ' '.join([w for w in sa_stopwords_lemma]) \
                     + '\t' \
                     + ' '.join([w for w in sb_stopwords_lemma]) \

            print(line, file=f_parse)
            print(s_word, file=f_word)
            print(s_lemma, file=f_lemma)
            print(s_pos, file=f_pos)
            print(s_ner, file=f_ner)
            print(s_stopwords_lemma, file=f_stopwords_lemma)

        f_parse.close()
        f_word.close()
        f_lemma.close()
        f_pos.close()
        f_ner.close()
        f_stopwords_lemma.close()

    ''' Load Data from File '''

    print('*' * 50)


    parse_data = []
    with codecs.open(parse_train_file, 'r', encoding='utf8') as f:
        for line in f:
            parse_json = json.loads(line)
            sentpair_instance = SentPair(parse_json)
            parse_data.append(sentpair_instance)

    print("Load Data, train_file=%s, gs_file=%s, n_train=%d\n" % (train_file, gs_file, len(parse_data)))
    return parse_data




if __name__ == '__main__':
    # s = "Snowden sees 'no chance' for US fair trial	Snowden sees \"no chance\" to get fair trial in U.S."
    # print(preprocess(s, 0.5))
    #
    # str = u'NA President‚Äôs message on the World Press Freedom Day	Pakistan marks World Press Freedom Day'
    # sa, sb = str.split('\t')
    # obj = nlp.parse(sa)
    # print(type(obj), obj)

    s = "On my own behalf and on behalf of my colleagues in the Committee on Fisheries, I would ask you, Madam President, to send Parliament' s condolences to the families of the victims and to the local authorities in both Brittany and in Marín, Galicia, from where the majority of the victims came.	Madam President, I would ask you, on behalf of my colleagues in the Committee on Fisheries and in my own name, to send a message of condolence on the part of Parliament to the families of the victims and the local authorities of Brittany as Marín, city of Galicia, where originating from most of the victims."
    sa, sb = s.split('\t')
    parse_sa = nlp.parse(sa)
    parse_sb = nlp.parse(sb)
    print("%s\n%s" % (sa, parse_sa))

    """
    print(json.dumps(parse_sa, indent=2))
    {
        "sentences": [
            {
                "tokens": [
                    {
                        "index": 1,
                        "word": "I",
                        "lemma": "I",
                        "after": " ",
                        "pos": "PRP",
                        "characterOffsetEnd": 1,
                        "characterOffsetBegin": 0,
                        "originalText": "I",
                        "ner": "O",
                        "before": ""
                    },
                    .....
                    {
                        "index": 8,
                        "word": ".",
                        "lemma": ".",
                        "after": "",
                        "pos": ".",
                        "characterOffsetEnd": 30,
                        "characterOffsetBegin": 29,
                        "originalText": ".",
                        "ner": "O",
                        "before": ""
                    }
                ],
                "basic-dependencies": [
                    {
                        "dep": "ROOT",
                        "dependent": 2,
                        "governorGloss": "ROOT",
                        "governor": 0,
                        "dependentGloss": "love"
                    },
                    ...
                    {
                        "dep": "punct",
                        "dependent": 8,
                        "governorGloss": "love",
                        "governor": 2,
                        "dependentGloss": "."
                    }
                ],
                "parse": "(ROOT\n  (S\n    (NP (PRP I))\n    (VP (VBP love)\n      (NP\n        (NP (NNP China) (. .))\n        (SBAR\n          (S\n            (NP (PRP I))\n            (VP (VBP love)\n              (NP (NNP Shanghai)))))))\n    (. .)))",
            }
        ]
    }
    """
