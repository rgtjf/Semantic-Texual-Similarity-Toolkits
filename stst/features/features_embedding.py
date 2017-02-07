from features import Feature
import nltk
import numpy as np
import pyjsonrpc
import dict_utils
import lib.kernel.vector_kernel as vk


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


class EmbeddingFeature(Feature):
    def __init__(self, emb_type, dim, pooling_type='avg', lower=True, **kwargs):
        super(EmbeddingFeature, self).__init__(**kwargs)

        self.lower = lower

        if 'emb_type' is None:
            print('please init with emb_type and dimension!')
            exit()
        self.emb_type = emb_type
        self.dim = dim
        self.pooling_type = pooling_type
        self.feature_name = self.feature_name + '%s-%s' % (emb_type, pooling_type)

    def extract(self, train_instance):
        lower = self.lower
        emb_type = self.emb_type
        dim = self.dim
        pooling_type = self.pooling_type

        word_sa, word_sb = train_instance.get_word(type='word', stopwords=True, lower=lower)

        pooling_vec_sa = pooling(word_sa, emb_type, dim, pooling_type)
        pooling_vec_sb = pooling(word_sb, emb_type, dim, pooling_type)
        all_feats, all_names = vk.get_all_kernel(pooling_vec_sa, pooling_vec_sb)
        features = all_feats

        infos = [emb_type, lower, pooling_type]
        return features, infos


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


class AverageEmbeddingSimilarityFeature(Feature):
    def sent2vec(self, word_sa, emb_type, dim, convey, lower=True):
        idf_weight = dict_utils.DictLoader().load_dict('idf')
        # idf_weight = dict_utils.DictLoader().load_idf_dict()

        embedding = Embedding()

        if lower == True:
            word_sa = [x.lower() for x in word_sa]

        vdist = nltk.FreqDist(word_sa)
        length = float(len(word_sa))

        vec = np.zeros((dim,), dtype=np.float32)
        # print(len(vec))
        for word in vdist:

            if convey == 'idf':
                w = idf_weight.get(word, 10.0)
            elif convey == 'tfidf':
                w = (vdist[word] / length) * idf_weight.get(word, 10.0)
            else:
                raise NotImplementedError

            if emb_type == 'word2vec':
                st, w2v = embedding.get_word2vec(word)
            elif emb_type == 'glove':
                st, w2v = embedding.get_glove(word)
            elif emb_type == 'paragram':
                st, w2v = embedding.get_paragram(word)
            elif emb_type == 'glove300':
                st, w2v = embedding.get_glove300(word)

            w2v = vk.normalize(w2v)

            w2v = vdist[word] * w * np.array(w2v)
            vec = vec + w2v
        # print(vec.dtype, vec.shape, len(vec))
        return vec

    def extract(self, train_instance):
        word_sa, word_sb = train_instance.get_word(type='word', stopwords=True, lower=True)
        features, infos = [], []

        if 'emb_type' not in self.kwargs:
            print('please init with emb_type and dimension!')
            exit()

        emb_type = self.kwargs['emb_type']
        # change feature_file according to their attribute
        self.feature_file = self.feature_file

        for lower in [True]:
            for convey in ['idf']:
                for emb_type, dim in [('paragram', 300), ('word2vec', 300),
                                      ('glove300', 300)]:  # , ('word2vec', 300) ('paragram', 300)
                    veca, vecb = self.sent2vec(word_sa, emb_type, dim, convey, lower), self.sent2vec(word_sb, emb_type,
                                                                                                     dim, convey, lower)
                    all_feats, all_names = vk.get_all_kernel(veca, vecb)
                    features += all_feats
                    infos.append([convey, emb_type, dim, lower, all_names])
        return features, infos


class MinEmbeddingSimilarityFeature(Feature):
    def sent2vec(self, word_sa, emb_type, dim, convey, lower=True):
        idf_weight = dict_utils.DictLoader().load_dict('idf')
        embedding = Embedding()

        if lower == True:
            word_sa = [x.lower() for x in word_sa]

        vdist = nltk.FreqDist(word_sa)
        length = float(len(word_sa))

        vec = []
        for word in vdist:

            if convey == 'idf':
                w = idf_weight.get(word, 10.0)

            if emb_type == 'word2vec':
                st, w2v = embedding.get_word2vec(word)
            elif emb_type == 'glove':
                st, w2v = embedding.get_glove(word)
            elif emb_type == 'paragram':
                st, w2v = embedding.get_paragram(word)
            w2v = vk.normalize(w2v)
            w2v = vdist[word] * w * np.array(w2v)
            vec.append(w2v)
        if len(vec) == 0:
            vec = np.zeros((dim,))
        else:
            vec = np.amin(vec, axis=0)
        return vec

    def extract(self, train_instance):
        word_sa, word_sb = train_instance.get_word(type='word', stopwords=True, lower=True)
        features, infos = [], []

        for lower in [True]:
            for convey in ['idf']:
                for emb_type, dim in [('paragram', 300), ('word2vec', 300)]:  # , ('word2vec', 300) ('paragram', 300)
                    veca, vecb = self.sent2vec(word_sa, emb_type, dim, convey, lower), self.sent2vec(word_sb, emb_type,
                                                                                                     dim, convey, lower)
                    all_feats, all_names = vk.get_all_kernel(veca, vecb)
                    features += all_feats
                    infos.append([convey, emb_type, dim, lower, all_names])
        return features, infos


class SelectedAverageEmbeddingSimilarityFeature(Feature):
    """
    kernal: 1 - 6
    """

    def sent2vec(self, word_sa, emb_type, dim, convey, lower=True):
        # idf_weight = dict_utils.DictLoader().load_dict('idf')
        idf_weight = dict_utils.DictLoader().load_idf_dict()

        embedding = Embedding()

        if lower == True:
            word_sa = [x.lower() for x in word_sa]

        vdist = nltk.FreqDist(word_sa)
        length = float(len(word_sa))

        vec = np.zeros((dim,), dtype=np.float32)
        # print(len(vec))
        for word in vdist:

            if convey == 'idf':
                w = idf_weight.get(word, 10.0)
            elif convey == 'tfidf':
                w = (vdist[word] / length) * idf_weight.get(word, 10.0)
            else:
                raise NotImplementedError

            if emb_type == 'word2vec':
                st, w2v = embedding.get_word2vec(word)
            elif emb_type == 'glove':
                st, w2v = embedding.get_glove(word)

            w2v = vk.normalize(w2v)

            w2v = vdist[word] * w * np.array(w2v)
            vec = vec + w2v
        # print(vec.dtype, vec.shape, len(vec))
        return vec

    def extract(self, train_instance):

        word_sa, word_sb = train_instance.get_word(type='word', stopwords=True, lower=True)

        features, infos = [], []

        for lower in [True]:
            for convey in ['tfidf', 'idf']:
                for emb_type, dim in [('word2vec', 300), ('glove', 100)]:
                    veca, vecb = self.sent2vec(word_sa, emb_type, dim, convey, lower), self.sent2vec(word_sb, emb_type,
                                                                                                     dim, convey, lower)
                    all_feats, all_names = vk.get_all_kernel(veca, vecb)
                    features += all_feats
                    infos.append([convey, emb_type, dim, lower, all_names])
        return features, infos


class MinMaxAvgW2VEmbeddingFeature(Feature):
    def extract(self, train_instance):
        word_sa, word_sb = train_instance.get_word(type='word', stopwords=True)

        # for emb_type, dim in (('word2vec', 300), ('glove', 100))[0]:
        emb_type, dim = 'word2vec', 300
        pooling_vec_sa, infos_sa = pooling(word_sa, emb_type, dim)
        pooling_vec_sb, infos_sb = pooling(word_sb, emb_type, dim)

        vec1 = pooling_vec_sa - pooling_vec_sb
        vec2 = pooling_vec_sa * pooling_vec_sb
        features = np.concatenate([vec1, vec2], axis=0)
        infos = [infos_sa, infos_sb]
        return features, infos

    def __test__(self):
        data = np.arange(6).reshape((3, 2))
        print(data)
        print(np.amin(data, axis=0))


class MinMaxAvgGloVeEmbeddingFeature(Feature):
    def extract(self, train_instance):
        word_sa, word_sb = train_instance.get_word(type='word', stopwords=True)

        # for emb_type, dim in (('word2vec', 300), ('glove', 100))[0]:
        emb_type, dim = 'glove', 100
        feature, info = pooling(word_sa, emb_type, dim)
        pooling_vec_sa, infos_sa = pooling(word_sa, emb_type, dim)
        pooling_vec_sb, infos_sb = pooling(word_sb, emb_type, dim)

        vec1 = pooling_vec_sa - pooling_vec_sb
        vec2 = pooling_vec_sa * pooling_vec_sb
        features = np.concatenate([vec1, vec2], axis=0)
        infos = [infos_sa, infos_sb]

        return features, infos

    def __test__(self):
        data = np.arange(6).reshape((3, 2))
        print(data)
        print(np.amin(data, axis=0))
