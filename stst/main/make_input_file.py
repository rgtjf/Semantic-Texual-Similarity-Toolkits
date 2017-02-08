# coding: utf8
from __future__ import print_function

from main_tools import *


def make(train_instance):
    sa, sb = train_instance.get_word(type='lemma', stopwords=False, lower=False)
    score = train_instance.get_score()
    return sa, sb, score


def load(train_file, train_gs=None, dev_flag=False):
    # train_file = config.TRAIN_FILE
    # train_gs = config.TRAIN_GS_FILE
    train_parse_data = data_utils.load_parse_data(train_file, train_gs, flag=dev_flag)
    datas = []
    for train_instance in train_parse_data:
        data = make(train_instance)
        datas.append(data)

    file_name = train_file.split('/')[-1]
    path_dir = '../iclr2016-test/data/eval/'
    f = utils.create_write_file(path_dir + file_name)
    for sa, sb, sc in datas:
        f.write('%s\t%s\t%.4f\n' % (' '.join(sa), ' '.join(sb), sc * 5))

    return datas


def main_en():
    train_file = config.TRAIN_FILE
    train_gs = config.TRAIN_GS_FILE
    load(train_file, train_gs)

    test_files = config.TEST_FILES['sts-en-en']
    task = 'sts-en-en'
    translator = 'manual'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')
        load(dev_file, dev_gs_file)

    test_files = config.TEST_FILES['STS2017.eval-sample']
    task = 'STS2017.eval'
    translator = 'manual'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))

        load(dev_file, None)

    test_files = config.TEST_FILES['sick']
    task = 'sts-en-en'
    translator = 'manual'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))

        load(dev_file, None)

def main_es(translator='googleapi'):
    task = 'sts-en-es'
    test_files = config.TEST_ES_FILES['sts-en-es']

    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input',translator.replace('microsoftapi','msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')
        load(dev_file, dev_gs_file)

    task = 'sts-es-es'
    test_files = config.TEST_ES_FILES['sts-es-es']
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')
        load(dev_file, dev_gs_file)

def main_test():
    """
    STS2017.eval-en': ['STS.input.track5.en-en.txt'],
    'STS2017.eval-snli': ['STS.input.track1.ar-ar.txt', 'STS.input.track2.ar-en.txt', 'STS.input.track3.es-es.txt', 'STS.input.track4a.es-en.txt', 'STS.input.track6.tr-en.txt'],
    'STS2017.eval-wmt': ['STS.input.track4b.es-en.txt'],
    :return:
    """
    test_files = config.TEST_FILES['STS2017.eval-en']
    task = 'STS2017.eval'
    translator = 'manual'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')
        load(dev_file, dev_gs_file)

    test_files = config.TEST_FILES['STS2017.eval-snli']
    translator = 'googleapi'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')
        load(dev_file, dev_gs_file)

    test_files = config.TEST_FILES['STS2017.eval-wmt']
    translator = 'googleapi_v2'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')
        load(dev_file, None)


def main_wmt():
    """
    """
    test_files = config.TEST_WMT_FILES['sts-en-es']
    task = 'sts-en-es'
    translator = 'googleapi_v2'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')
        load(dev_file, dev_gs_file)


    test_files = config.TEST_FILES['STS2017.eval-wmt']
    task = 'STS2017.eval'
    translator = 'googleapi_v2'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')
        load(dev_file, None)

if __name__ == '__main__':
    main_wmt()