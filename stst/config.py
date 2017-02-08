
# """
#   File Config
# """
#
# RESOURCE_DIR = ROOT
# TRAIN_DIR = RESOURCE_DIR + '/data'
#
# train = ['all']
# TRAIN_FILE = TRAIN_DIR + '/sts-en-en/input.%s.txt' % (train[1])
#
# TEST_FILES = {
#     'sts-en-en': ['STS2016.input.answer-answer.txt', 'STS2016.input.headlines.txt', 'STS2016.input.plagiarism.txt',
#                   'STS2016.input.postediting.txt', 'STS2016.input.question-question.txt']
# }
#
GENERATE = '../generate'

''' result out '''
OUTPUT_DIR = GENERATE + '/outputs'

''' feature config '''
FEATURE_DIR = GENERATE + '/features'

''' model config '''
MODEL_DIR = GENERATE + '/models'

''' record config '''
RECORD_DIR = GENERATE + '/records'


RESOURCE = 'resources'
''' dict config '''
DICT_DIR = RESOURCE

#
# ''' nn feature config '''
# NN_FEATURE_PATH = RESOURCE + '/iclr2016-test/data/features'

