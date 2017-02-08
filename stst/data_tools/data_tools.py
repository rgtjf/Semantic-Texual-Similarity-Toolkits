import stst.config
from stst.data_tools import data_utils


def get_all_instance(file_list, type='lemma'):
    """
    sentence_dict['file'][idx]['sa'] = idx
    sentence_dict['file'][idx]['sb'] = idx+1
    """
    sentence_tags = []
    sentences = []
    for file in file_list:
        # file is path
        file_name = file.split('/')[-1]
        parse_data = data_utils.load_parse_data(file)
        for idx, train_instance in enumerate(parse_data):
            if type == 'lemma':
                sa, sb = train_instance.get_word(type='lemma', stopwords=False, lower=True)
            elif type == 'word' :
                sa, sb = train_instance.get_word(type='word')
            sa_tag = "%s_%d_sa" % (file_name, idx)
            sb_tag = "%s_%d_sb" % (file_name, idx)

            sentences.append(sa)
            sentence_tags.append(sa_tag)

            sentences.append(sb)
            sentence_tags.append(sb_tag)

    return sentences, sentence_tags


def get_sts_file_list():
    file_list = []

    # train_file
    train_file = stst.config.TRAIN_FILE
    file_list.append(train_file)

    # sts-en-en
    task = 'sts-en-en'
    test_files = stst.config.TEST_FILES['sts-en-en']
    translator = 'manual'

    for corpus in test_files:
        dev_file = stst.config.TEST_DIR + '/' + task + '/' + translator + '/' + corpus.replace('input', translator.replace(
        'microsoftapi', 'msapi'))
        file_list.append(dev_file)

    # STS2017.eval-sample
    task = 'STS2017.eval'
    translator = 'manual'
    test_files = stst.config.TEST_FILES['STS2017.eval-sample']
    for corpus in test_files:
        dev_file = stst.config.TEST_DIR + '/' + task + '/' + translator + '/' + corpus.replace('input', translator.replace(
            'microsoftapi', 'msapi'))
        file_list.append(dev_file)

    #
    task = 'STS2017.eval'
    translator = 'manual'
    test_files = stst.config.TEST_FILES['STS2017.eval-en']
    for corpus in test_files:
        dev_file = stst.config.TEST_DIR + '/' + task + '/' \
                   + translator + '/' + corpus.replace('input',translator.replace('microsoftapi', 'msapi'))
        file_list.append(dev_file)

    task = 'STS2017.eval'
    translator = 'googleapi'
    test_files = stst.config.TEST_FILES['STS2017.eval-snli']
    for corpus in test_files:
        dev_file = stst.config.TEST_DIR + '/' + task + '/' \
                   + translator + '/' + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        file_list.append(dev_file)

    task = 'STS2017.eval'
    translator = 'googleapi_v2'
    test_files = stst.config.TEST_FILES['STS2017.eval-wmt']
    for corpus in test_files:
        dev_file = stst.config.TEST_DIR + '/' + task + '/' \
                   + translator + '/' + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        file_list.append(dev_file)

    # print('\n'.join(file_list))
    #
    # print('\n'.join([file.split('/')[-1] for file in file_list]))

    return file_list

def get_wmt_file_list():
    file_list = []

    task = 'sts-en-es'
    test_files = stst.config.TEST_WMT_FILES['sts-en-es']
    translator = 'googleapi_v2'

    for corpus in test_files:
        dev_file = stst.config.TEST_DIR + '/' + task + '/' + translator + '/' + corpus.replace('input', translator.replace(
        'microsoftapi', 'msapi'))
        file_list.append(dev_file)


    task = 'STS2017.eval'
    translator = 'googleapi_v2'
    test_files = stst.config.TEST_FILES['STS2017.eval-wmt']
    for corpus in test_files:
        dev_file = stst.config.TEST_DIR + '/' + task + '/' \
                   + translator + '/' + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
        file_list.append(dev_file)

    return file_list