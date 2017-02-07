ROOT = '..'

"""
  File Config
"""

RESOURCE_DIR = ROOT + '/resources'

TRAIN_DIR = RESOURCE_DIR + '/data'

train = ['all']
TRAIN_FILE = TRAIN_DIR + '/sts-en-en/input.%s.txt' % (train[1])
TRAIN_GS_FILE = TRAIN_DIR + '/sts-en-en/gs.%s.txt' % (train[1])


dev = ['news', 'multisource'][0]
DEV_DIR = RESOURCE_DIR + '/data'
DEV_FILE = DEV_DIR + '/sts-en-es/googleapi/STS.googleapi.%s.txt' % (dev)
DEV_GS_FILE = DEV_DIR + '/sts-en-es/STS.gs.%s.txt' % (dev)



"""
sts-en-es
sts-ar-ar
sts-en-ar
sts-es-es
"""
TEST_FILES = {
    'STS2017.eval-sample': ['STS.manual.track5-sample.en-en.txt'],
    'STS2017.eval-en': ['STS.input.track5.en-en.txt'],
    'STS2017.eval-snli': ['STS.input.track1.ar-ar.txt', 'STS.input.track2.ar-en.txt', 'STS.input.track3.es-es.txt', 'STS.input.track4a.es-en.txt', 'STS.input.track6.tr-en.txt'],
    'STS2017.eval-wmt': ['STS.input.track4b.es-en.txt'],
    'sick': ['SICK_trial.manual.SICK.txt'],
    'sts-en-en': ['STS2016.input.answer-answer.txt', 'STS2016.input.headlines.txt', 'STS2016.input.plagiarism.txt',
                 'STS2016.input.postediting.txt', 'STS2016.input.question-question.txt']
}

TEST_ES_FILES = {
    'sts-en-es': ['STS.input.wmt.txt', 'STS.input.news.txt', 'STS.input.multisource.txt'],
    'sts-es-es': ['sts.2014.input.li65.txt', 'STS.2014.input.news.txt', 'STS.2014.input.wikipedia.txt',
                  'STS.2015.input.newswire.txt', 'STS.2015.input.wikipedia.txt']
}

TEST_WMT_FILES = {
    'sts-en-es': ['STS.input.wmt.txt']
}


TEST_DIR = RESOURCE_DIR + '/data'
TEST_FILE = TEST_DIR + '/sts-en-es/googleapi/STS.googleapi.news.txt'
TEST_GS_FILE = TEST_DIR + '/sts-en-es/STS.gs.news.txt'


"""
    Feature Config
"""

NN_FEATURE_PATH = ROOT + '/iclr2016-test/data/features'

''' result out '''
OUTPUT_DIR = ROOT + '/outputs'


''' dict config '''
DICT_DIR = RESOURCE_DIR + '/editable_dict'
EX_DICT_DIR = RESOURCE_DIR + '/external_dict'


''' feature config '''
FEARURE_DIR = ROOT + '/features'

''' model config '''
MODEL_DIR = ROOT + '/models'

''' record config '''
RECORD_DIR = ROOT + '/records'

