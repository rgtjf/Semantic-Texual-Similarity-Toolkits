# coding: utf8

from __future__ import print_function

import json
import codecs
import scipy.stats as meas
import numpy as np

import stst
from stst import utils
import corenlp_utils

def stanford_tokenize(s):
    return corenlp_utils.tokenize(s, type='lemma')


class DefineExample(stst.Example):

    def __init__(self, example_dict):
        """
        sa, sb: word list
        score: float
        """
        self.sa = example_dict['sa'].lower().split()
        self.sb = example_dict['sb'].lower().split()
        self.en_sa = example_dict['en_sa']
        self.en_sb = example_dict['en_sb']
        self.score = example_dict['score']

        self.translete = False
        self.token_en_sa = None
        self.token_en_sb = None

    def get_words(self):
        """ Return sa, sb """
        return self.sa, self.sb

    def get_en_words(self):
        if self.translete is False:
            sa = stanford_tokenize(self.en_sa)
            sa = [w.lower() for w in sa]
            sb = stanford_tokenize(self.en_sb)
            sb = [w.lower() for w in sb]
            self.token_en_sa = sa
            self.token_en_sb = sb
        else:
            sa = self.token_en_sa
            sb = self.token_en_sb
        return sa, sb

    def get_score(self):
        """ Return the gold score """
        return self.score

    def get_instance_string(self):
        """ Return instance string """
        instance_string = "{}\t{}\t{}".format(self.score, self.en_sa, self.en_sb)
        return instance_string

    @staticmethod
    def load_data(file_path):
        """ Return list of examples """
        examples = []
        with utils.create_read_file(file_path, encoding='utf8') as f:
            for line in f:
                example_dict = json.loads(line.strip())
                example = DefineExample(example_dict)
                examples.append(example)
        return examples


class DefineEvaluation(stst.Evaluation):

    def __init__(self, predicts, golds):
        self.predicts = predicts
        self.golds = golds

    def evaluation(self):
        result = {}
        result['pearsonr'] = self.pearsonr(self.predicts, self.golds)
        return result

    @staticmethod
    def pearsonr(predict, gold):
        """
        pearsonr of predict and gold
        :param predict: list
        :param gold: list
        :return: mape
        """
        pearsonr = meas.pearsonr(predict, gold)[0]
        return pearsonr

    @staticmethod
    def init_from_modelfile(output_file):
        """
        output_file:
            FORMAT
                25.00   #   1.8 A woman is cutting onions.  A woman is cutting tofu.
        """
        predicts, golds = [], []
        with open(output_file) as f:
            for line in f:
                items = line.strip().split('\t#\t')
                predict = float(items[0])
                gold = float(items[1].split()[0])
                predicts.append(predict)
                golds.append(gold)
        return DefineEvaluation(predicts, golds)


class DefineFeature(stst.Feature):
    def extract(self, train_instance):
        sa, sb = train_instance.get_words()
        features = [abs(len(sa) - len(sb))]
        infos = ['test']
        return features, infos


class OverlapFeature(stst.Feature):
    def extract(self, train_instance):
        sa, sb = train_instance.get_words()
        f1 = utils.ngram_match(sa, sb, 1)
        info = [sa, sb]
        return [f1], info



def pooling(word_embs, dim, pooling_types):
    if pooling_types == 'avg':
        function = np.average
    elif pooling_types == 'min':
        function = np.amin
    elif pooling_types == 'max':
        function = np.amax
    else:
        print(pooling_types)
        raise NotImplementedError

    if len(word_embs) == 0:
        vec = np.zeros((dim,))
    else:
        vec = function(word_embs, axis=0)
    return vec


def minavgmaxpooling(word_list, w2i, embeddings, dim, convey, idf_weight):
    word_embs = []
    for word in word_list:
        # gain word vector
        w2v = embeddings[w2i[word]]

        # gain weight of word
        default_idf_weight = min(idf_weight.values())
        if convey == 'idf':
            w = idf_weight.get(word, default_idf_weight)
        elif convey == 'count':
            w = 1.0
        else:
            raise NotImplementedError

        # append
        w2v = w * np.array(w2v)
        word_embs.append(w2v)

    # concat sentence embedding
    vecs = []
    for pooling_types in ['avg']: # ['avg', 'min', 'max']:
        vec = pooling(word_embs, dim, pooling_types)
        vecs.append(vec)
    vecs = np.reshape(vecs, [-1])
    return vecs


class CLMinAvgMaxEmbeddingFeature(stst.Feature):
    def __init__(self, emb_name, dim, ar_emb_file, en_emb_file, trans_file=None, binary=False, **kwargs):
        super(CLMinAvgMaxEmbeddingFeature, self).__init__(**kwargs)
        self.emb_name = emb_name
        self.dim = dim
        self.ar_emb_file = ar_emb_file
        self.en_emb_file = en_emb_file
        self.trans_file = trans_file

        self.binary = binary
        self.word_type = 'word'
        self.feature_name = self.feature_name + '-%s' % (emb_name)

    def extract_information(self, train_instances):
        en_seqs = []
        ar_seqs = []
        # (sa, sb): (en, ar)
        # (sa, sb): (en, es)
        for train_instance in train_instances:
            word_sa, word_sb = train_instance.get_words()
            en_seqs.append(word_sa)
            ar_seqs.append(word_sb)

        self.en_idf_weight = utils.idf_calculator(en_seqs)
        self.en_word2index = {word: index + 2 for index, word in enumerate(self.en_idf_weight)}
        self.en_word2index['__PAD__'] = 0
        self.en_word2index['__UNK__'] = 1
        self.en_embeddings = utils.load_word_embedding(self.en_word2index, self.en_emb_file, self.dim)

        self.ar_idf_weight = utils.idf_calculator(ar_seqs)
        self.ar_word2index = {word: index + 2 for index, word in enumerate(self.ar_idf_weight)}
        self.ar_word2index['__PAD__'] = 0
        self.ar_word2index['__UNK__'] = 1
        self.ar_embeddings = utils.load_word_embedding(self.ar_word2index, self.ar_emb_file, self.dim)

        if self.trans_file:
            transmat = np.loadtxt(self.trans_file)
            self.ar_embeddings = np.matmul(self.ar_embeddings, transmat).astype(np.float32)



    def extract(self, train_instance):

        word_sa, word_sb = train_instance.get_words()

        pooling_vec_sa = minavgmaxpooling(word_sa, self.en_word2index, self.en_embeddings, self.dim,
                                          convey='idf', idf_weight=self.en_idf_weight)
        pooling_vec_sb = minavgmaxpooling(word_sb, self.ar_word2index, self.ar_embeddings, self.dim,
                                          convey='idf', idf_weight=self.ar_idf_weight)

        features = [ 1 - utils.cosine_distance(pooling_vec_sa, pooling_vec_sb, norm=True)]
        infos = [self.emb_name, self.word_type]
        return features, infos


class CLCharNgramFeature(stst.Feature):

    @staticmethod
    def char_gram(text, n=3):
        if type(text) is list:
            text = ' '.join(text)
        text = text.lower()
        text = list(filter(lambda c: c.isalnum(),text))
        ngram = utils.make_ngram(text, n)
        return ngram

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            word_sa, word_sb = train_instance.get_words()
            char_ngram_sa = self.char_gram(word_sa, 3)
            char_ngram_sb = self.char_gram(word_sb, 3)
            seqs.append(char_ngram_sa)
            seqs.append(char_ngram_sb)

        self.idf_weight = utils.idf_calculator(seqs)
        self.vocab = utils.word2index(self.idf_weight)

    def extract(self, tran_instance):
        word_sa, word_sb = tran_instance.get_words()
        char_ngram_sa = self.char_gram(word_sa, 3)
        char_ngram_sb = self.char_gram(word_sb, 3)

        vec1 = utils.vectorize(char_ngram_sa, self.idf_weight, self.vocab, convey='idf')
        vec2 = utils.vectorize(char_ngram_sb, self.idf_weight, self.vocab, convey='idf')
        features, info = [1 - utils.cosine_distance(vec1, vec2, norm=True)], 'cosine'
        return features, info


class MTMinAvgMaxEmbeddingFeature(stst.Feature):
    def __init__(self, emb_name, dim, ar_emb_file, en_emb_file, trans_file=None, binary=False, **kwargs):
        super(MTMinAvgMaxEmbeddingFeature, self).__init__(**kwargs)
        self.emb_name = emb_name
        self.dim = dim
        self.ar_emb_file = ar_emb_file
        self.en_emb_file = en_emb_file
        self.trans_file = trans_file

        self.binary = binary
        self.word_type = 'word'
        self.feature_name = self.feature_name + '-%s' % (emb_name)

    def extract_information(self, train_instances):
        en_seqs = []
        # (sa, sb): (en, ar)
        # (sa, sb): (en, es)
        for train_instance in train_instances:
            word_sa, word_sb = train_instance.get_en_words()
            en_seqs.append(word_sa)
            en_seqs.append(word_sb)

        self.en_idf_weight = utils.idf_calculator(en_seqs)
        self.en_word2index = {word: index + 2 for index, word in enumerate(self.en_idf_weight)}
        self.en_word2index['__PAD__'] = 0
        self.en_word2index['__UNK__'] = 1
        self.en_embeddings = utils.load_word_embedding(self.en_word2index, self.en_emb_file, self.dim)

    def extract(self, train_instance):

        word_sa, word_sb = train_instance.get_en_words()

        pooling_vec_sa = minavgmaxpooling(word_sa, self.en_word2index, self.en_embeddings, self.dim,
                                          convey='idf', idf_weight=self.en_idf_weight)
        pooling_vec_sb = minavgmaxpooling(word_sb, self.en_word2index, self.en_embeddings, self.dim,
                                          convey='idf', idf_weight=self.en_idf_weight)

        features = [ 1 - utils.cosine_distance(pooling_vec_sa, pooling_vec_sb, norm=True)]
        infos = [self.emb_name, self.word_type]
        return features, infos