# coding: utf8
import cparams

from stst.evaluation import *
from stst.record import *
from stst.data_tools import data_utils


def make_features(model):
    test_files = config.TEST_FILES
    for task in test_files:
        for corpus in test_files[task]:
            translators = ['googleapi', 'microsoftapi']
            if 'ar' in task:
                translators.append('manual')
            if 'en-en' in task:
                translators = ['manual']
            for translator in translators:
                dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                           + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
                dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
                dev_gs_file = dev_gs_file.replace('input', 'gs')
                dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=False)
                model.make_feature_file(dev_parse_data, dev_file)


def train_en(model, out_list=None):
    if cparams.train_sick:
        train_file = config.SICK_TRAIN_FILE
        train_gs = train_file
    else:
        train_file = config.TRAIN_FILE
        train_gs = config.TRAIN_GS_FILE

    ''' Build SentPair Object'''
    train_parse_data = data_utils.load_parse_data(train_file, train_gs, flag=False)
    out_list = cparams.train_out_list
    model.train(train_parse_data, train_file, out_list)


def test_en(model):
    if cparams.test_sick:
        test_files = [ config.TEST_FILES['sick'] ]
    else:
        test_files = config.TEST_FILES['sts-en-en']
    task = 'sts-en-en'
    translator = 'manual'
    output_file_list, gs_file_list = [], []
    info = {}
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')
        dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=False, sick=cparams.test_sick)
        model.test(dev_parse_data, dev_file)

        gs_file_list.append(dev_gs_file)
        output_file_list.append(model.output_file)

        pearsonr = eval_file(model.output_file, dev_gs_file)
        info[corpus] = pearsonr
        print(task, pearsonr, translator, corpus)

    pearsonr = eval_file_corpus(output_file_list, gs_file_list)
    print(task, pearsonr, translator)
    record_corpus('input-all', task, pearsonr, translator, model, info)
    return pearsonr, info


def predict_en_sample(model, dev_flag=False):

    test_files = config.TEST_FILES['STS2017.eval-sample']
    task = 'STS2017.eval'
    translator = 'manual'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' + corpus.replace('input', translator.replace(
            'microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('manual', 'gs')
        dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=dev_flag)
        model.test(dev_parse_data, dev_file)
        pearsonr = eval_file(model.output_file, dev_gs_file)
        print(task, pearsonr, translator, corpus)
    print("===> Predict STS2017.eval-sample finished")


def predict_en(model, dev_flag=False):

    test_files = config.TEST_FILES['STS2017.eval-en']
    task = 'STS2017.eval'
    translator = 'manual'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' + corpus.replace('input', translator.replace(
            'microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('manual', 'gs')
        dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=dev_flag)
        model.test(dev_parse_data, dev_file)



def predict_snli(model, translator='googleapi', dev_flag=False):
    test_files = config.TEST_FILES['STS2017.eval-snli']
    task = 'STS2017.eval'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' + corpus.replace('input', translator.replace(
            'microsoftapi', 'msapi'))
        dev_gs_file = None
        dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=dev_flag)
        model.test(dev_parse_data, dev_file)


def predict_wmt(model, translator='googleapi_v2', dev_flag=False):
    test_files = config.TEST_FILES['STS2017.eval-wmt']
    task = 'STS2017.eval'
    for corpus in test_files:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' + corpus.replace('input', translator.replace(
            'microsoftapi', 'msapi'))
        dev_gs_file = None
        dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=dev_flag)
        model.test(dev_parse_data, dev_file)

def test_en_es(model, translator='googleapi', dev_flag=False):
    test_files = config.TEST_ES_FILES
    # for task in test_files:
    info = {}
    task = 'sts-en-es'
    for corpus in test_files[task]:
        # if 'news' in corpus:
        #     continue
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')

        dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=dev_flag)

        model.test(dev_parse_data, dev_file)
        pearsonr = eval_file(model.output_file, dev_gs_file)
        info[corpus] = pearsonr
        print(task, pearsonr, translator, corpus)
    record_other_corpus('input-all', task, translator, model, info)


def train_wmt(model):
    ''' Build SentPair Object'''
    test_files = config.TEST_WMT_FILES
    task = 'sts-en-es'
    info = {}
    corpus = test_files[task][0]
    translator = 'googleapi_v2'
    train_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
               + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
    train_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
    train_gs_file = train_gs_file.replace('input', 'gs')
    train_parse_data = data_utils.load_parse_data(train_file, train_gs_file, flag=False)
    model.train(train_parse_data, train_file)

def test_wmt(model, translator='googleapi_v2', dev_flag=False):
    test_files = config.TEST_WMT_FILES
    task = 'sts-en-es'
    info = {}
    for corpus in test_files[task]:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')

        dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=dev_flag)

        model.test(dev_parse_data, dev_file)
        pearsonr = eval_file(model.output_file, dev_gs_file)
        info[corpus] = pearsonr
        print(task, pearsonr, translator, corpus)
    record_other_corpus('input-all', task, translator, model, info)
    return pearsonr, info


def cv_test_wmt(model, translator='googleapi_v2', dev_flag=False):
    test_files = config.TEST_WMT_FILES
    task = 'sts-en-es'
    info = {}
    for corpus in test_files[task]:
        dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                   + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
        dev_gs_file = dev_gs_file.replace('input', 'gs')

        dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=dev_flag)

        model.cross_validation(dev_parse_data, dev_file)


def test_ar(model, translator='googleapi'):
    """
    re train the model
    """
    test_files = config.TEST_AR_FILES
    for task in test_files:
        info = {}
        for corpus in test_files[task]:
            dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                       + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
            dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
            dev_gs_file = dev_gs_file.replace('input', 'gs')

            dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=False)
            model.test(dev_parse_data, dev_file)

            pearsonr = eval_file(model.output_file, dev_gs_file)
            info[corpus] = pearsonr
            print(task, pearsonr, translator, corpus)
        record_other_corpus('input-all', task, translator, model, info)


def hill_climbing_en(model, choose_list=[]):


    chooses = choose_list

    feature_list= model.feature_list
    visited = [True if x in choose_list else False for x in range(len(feature_list))]

    for idx in range(len(choose_list), len(feature_list)):
        chooseIndex = -1
        best_score = 0.0
        chooses.append(-1)
        for i in range(len(feature_list)):
            if visited[i] == False:
                chooses[idx] = i
                feature = [feature_list[s] for s in chooses]
                # print(len(feature_list))
                model.feature_list = feature
                train_en(model)
                cur_score, info = test_en(model)
                if best_score < cur_score:
                    chooseIndex = i
                    best_score = cur_score
        chooses[idx] = chooseIndex
        visited[chooseIndex] = True
        # feature = [ feature_list[s] for s in chooses]
        print('Best Score: %.2f %% ,choose Feature %s' % (best_score * 100, feature_list[chooseIndex].feature_name))


def hill_climbing(model, test_func, choose_list=[]):


    chooses = choose_list

    feature_list= model.feature_list
    visited = [True if x in choose_list else False for x in range(len(feature_list))]

    for idx in range(len(choose_list), len(feature_list)):
        chooseIndex = -1
        best_score = 0.0
        chooses.append(-1)
        for i in range(len(feature_list)):
            if visited[i] == False:
                chooses[idx] = i
                feature = [feature_list[s] for s in chooses]
                # print(len(feature_list))
                model.feature_list = feature
                train_en(model)
                cur_score, info = test_func(model)
                if best_score < cur_score:
                    chooseIndex = i
                    best_score = cur_score
        chooses[idx] = chooseIndex
        visited[chooseIndex] = True
        # feature = [ feature_list[s] for s in chooses]
        print('Best Score: %.2f %% ,choose Feature %s' % (best_score * 100, feature_list[chooseIndex].feature_name))


if __name__ == '__main__':
    pass

    # for key in config.TEST_FILES:
    #     print(key)

    # parse_files()
