import nltk
import numpy as np
import pyjsonrpc
from features import Feature

from stst.data import dict_utils
from stst.libs.kernel import vector_kernel as vk


class Embedding(object):
    def __init__(self):
        self.http_client = pyjsonrpc.HttpClient(
            url="http://localhost:8084",
        )

    def get_word2vec(self, word):
        """

        :param word:
        :return: (st, vec)
        """
        vec = self.http_client.word2vec(word)
        return vec

    def get_glove(self, word):
        vec = self.http_client.glove(word)
        return vec

    def get_paragram(self, word):
        vec = self.http_client.paragram(word)
        return vec

    def get_glove300(self, word):
        vec = self.http_client.glove300(word)
        return vec


def pooling(word_sa, emb_type, dim, pooling_types='avg', convey='idf'):
    idf_weight = dict_utils.DictLoader().load_dict('idf')
    embedding = Embedding()

    vdist = nltk.FreqDist(word_sa)
    length = float(len(word_sa))

    if pooling_types == 'avg':
        function = np.average
    elif pooling_types == 'min':
        function = np.amin
    elif pooling_types == 'max':
        function = np.amax
    else:
        print(pooling_types)
        raise NotImplementedError


    vec = []
    for word in word_sa:
        if emb_type == 'word2vec':
            st, w2v = embedding.get_word2vec(word)
        elif emb_type == 'glove':
            st, w2v = embedding.get_glove(word)
        elif emb_type == 'paragram':
            st, w2v = embedding.get_paragram(word)
        elif emb_type == 'glove300':
            st, w2v = embedding.get_glove300(word)

        if convey == 'idf':
            w = idf_weight.get(word, 10.0)
        elif convey == 'tfidf':
            w = vdist[word] * idf_weight.get(word, 10.0)
        else:
            raise NotImplementedError

        w2v = w * np.array(w2v)
        vec.append(w2v)

    if len(vec) == 0:
        vec = np.zeros((dim,))
    else:
        vec = function(vec, axis=0)

    return vec


def minavgmaxpooling(word_sa, emb_type, dim, convey='idf'):
    vecs = []
    for pooling_types in ['avg', 'min', 'max']:
        vec = pooling(word_sa, emb_type, dim, pooling_types, convey)
        vecs.append(vec)

    vecs = np.reshape(vecs, [-1])
    return vecs


class MinAvgMaxEmbeddingFeature(Feature):
    def __init__(self, emb_type, dim,  lower=True, **kwargs):
        super(MinAvgMaxEmbeddingFeature, self).__init__(**kwargs)

        self.lower = lower

        if 'emb_type' is None:
            print('please init with emb_type and dimension!')
            exit()
        self.emb_type = emb_type
        self.dim = dim
        self.feature_name = self.feature_name + '-%s' % (emb_type)

    def extract(self, train_instance):
        lower = self.lower
        emb_type = self.emb_type
        dim = self.dim

        word_sa, word_sb = train_instance.get_word(type='word', stopwords=True, lower=lower)

        pooling_vec_sa = minavgmaxpooling(word_sa, emb_type, dim)
        pooling_vec_sb = minavgmaxpooling(word_sb, emb_type, dim)
        all_feats, all_names = vk.get_all_kernel(pooling_vec_sa, pooling_vec_sb)
        features = all_feats

        infos = [emb_type, lower]
        return features, infos


class MinAvgMaxPoolingFeature(Feature):
    def __init__(self, emb_name, dim, emb_file, binary=False, lower=True, **kwargs):
        pass
