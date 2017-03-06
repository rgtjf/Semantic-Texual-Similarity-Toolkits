from stst import dict_utils, utils
from stst.data_tools import data_utils

from stst import config


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


def load_STS():
    train_file = config.STS_TRAIN_FILE
    dev_file = config.STS_DEV_FILE
    test_file = config.STS_TEST_FILE
    load(train_file)
    load(dev_file)
    load(test_file)


load_STS()