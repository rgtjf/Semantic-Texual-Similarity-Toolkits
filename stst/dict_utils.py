# coding: utf8
from __future__ import print_function

import math
import os

import stst
import stst.config
from stst import utils, config


@utils.singleton
class DictLoader(object):
    def __init__(self):
        self.dict_manager = {}

    def     load_dict(self, dict_name, path=config.DICT_DIR):
        """
        path: config.DICT_DIR
              config.DICT_EX_DIR
        """
        if dict_name not in self.dict_manager:

            dict_object = {}

            cur_dir = os.path.dirname(__file__)
            path = os.path.join(cur_dir, 'resources')

            ''' load dict from file '''
            file_name = path + '/dict_%s.txt' % dict_name
            print('load dict from file %s \n' % file_name)

            f_dict = utils.create_read_file(file_name)

            for idx, line in enumerate(f_dict):
                line = line.strip().split('\t')
                if len(line) == 1:
                    dict_object[line[0]] = idx + 1
                elif len(line) == 2:
                    dict_object[line[0]] = eval(line[1])
                else:
                    raise NotImplementedError

            self.dict_manager[dict_name] = dict_object

        return self.dict_manager[dict_name]

    def load_doc2vec(self):
        dict_name = 'doc2vec'
        if dict_name not in self.dict_manager:
            from gensim.models import Doc2Vec
            model = Doc2Vec.load(config.EX_DICT_DIR + '/doc2vec.model')
            self.dict_manager[dict_name] = model
        return self.dict_manager[dict_name]

    def load_idf_dict(self, dict_name='idf_dict'):

        if dict_name not in self.dict_manager:

            word_frequencies = {}

            file_name = config.EX_DICT_DIR + '/word-frequencies.txt'
            print('load dict from file %s \n' % file_name)

            f_dict = utils.create_read_file(file_name)

            for idx, line in enumerate(f_dict):
                if idx == 0:
                    totfreq = int(line)
                else:
                    w, freq = line.strip().split()
                    freq = float(freq)
                    if freq < 10:
                        continue
                    word_frequencies[w] = math.log(totfreq / freq)  / math.log(2)
            self.dict_manager[dict_name] = word_frequencies

        return self.dict_manager[dict_name]


class DictCreater(object):
    def __init__(self):
        pass

    def _create_dict(func):
        """
        Python 装饰器
        """

        def __create_dict(*args, **kwargs):
            print("==" * 40)
            print("Create dict for %s ...  " % (func.__name__.replace("create_", "")))
            print("==" * 40)
            ret = func(*args, **kwargs)

            ''' remove item whose frequency is less than threshold '''
            if 'threshold' in kwargs:
                threshold = kwargs['threshold']
                for key in ret.keys():
                    if ret[key] < threshold:
                        ret.pop(key)

            ''' write dict to file '''

            file_name = 'dict_' + func.__name__.replace("create_", "") + '.txt'
            f_dict = utils.create_write_file(config.DICT_DIR + '/' + file_name)

            if type(ret) == list:
                # ensure it is set
                ret = list(set(ret))
                ret = sorted(ret)
                for idx, item in enumerate(ret):
                    print(str(item), file=f_dict)

            elif type(ret) == dict:
                # order the dict
                for item in sorted(ret.keys()):
                    print(item, ret[item])
                    print('%s\t%s' % (item, ret[item]), file=f_dict)
            else:
                raise NotImplementedError

            f_dict.close()

            print("Write file: %s, %d instances" % (file_name, len(ret)))
            return ret

        return __create_dict

    @_create_dict
    def create_stopwords(self):
        # stopwords 2: zhaojiang's
        # fp = open(config.DICT_DIR + '/english.stopwords.txt', 'r')
        # english_stopwords = [line.strip('\r\n ') for line in fp.readlines()]

        # stopwords 3: stanford
        fp = open(config.DICT_DIR + '/stanford.stopwords.txt', 'r')
        stanford_stopwords = [line.strip('\r\n ') for line in fp.readlines()]

        stopwords = stanford_stopwords
        return stopwords


    @_create_dict
    def create_global_idf(self, file_list):
        print('\n'.join(file_list))
        sentences, _ = stst.load_sentences(file_list)
        print(sentences[:5])
        global_idf = utils.idf_calculator(sentences)
        return global_idf


if __name__ == '__main__':


    # import data_utils
    #
    # ''' Load SentPair Object'''
    # train_file = config.TRAIN_FILE
    # train_gs = config.TRAIN_GS_FILE
    # train_parse_data = data_utils.load_parse_data(train_file, train_gs, flag=False)
    #
    # sentences = []
    # for train_instance in train_parse_data:
    #     sa, sb = train_instance.get_word(type='word')
    #     sentences.append(sa)
    #     sentences.append(sb)
    #
    # DictCreater().create_topic(sentences)
    DictCreater().create_global_idf()  # lemma, lower=True, stopwords=False
    # DictCreater().create_corpus_idf()
    # DictLoader().load_dict('stopwords')
    # DictCreater().create_stopwords()

