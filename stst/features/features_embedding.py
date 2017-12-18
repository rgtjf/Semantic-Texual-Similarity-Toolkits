# coding: utf8
import nltk
import numpy as np

from stst.modules.features import Feature
from stst.libs.kernel import vector_kernel as vk
from stst import utils


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


def minavgmaxpooling(word_list, stoi, embeddings, dim, convey, idf_weight):

    word_embs = []
    min_idf_weight = min(idf_weight.values())

    for word in word_list:
        if word not in stoi:
            continue

        # gain word vector
        w2v = embeddings[stoi[word]]

        if convey == 'idf':
            w = idf_weight.get(word, min_idf_weight)
        else:
            raise NotImplementedError

        # append
        w2v = w * np.array(w2v)
        word_embs.append(w2v)

    # concat sentence embedding
    vecs = []
    for pooling_types in ['avg', 'min', 'max']:
        vec = pooling(word_embs, dim, pooling_types)
        vecs.append(vec)
    vecs = np.reshape(vecs, [-1])
    return vecs


class MinAvgMaxEmbeddingFeature(Feature):

    def __init__(self, emb_name, dim, emb_file, word_type, **kwargs):
        super(MinAvgMaxEmbeddingFeature, self).__init__(**kwargs)
        self.emb_name = emb_name
        self.dim = dim
        self.emb_file = emb_file

        self.word_type = word_type
        self.stopwords = True
        self.lower = True
        self.binary = True if 'bin' in emb_file else False
       
        self.feature_name += '-%s-%s' % (emb_name, word_type)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            word_sa, word_sb = train_instance.get_word(type=self.word_type,
                                                       stopwords=self.stopwords,
                                                       lower=self.lower)
            seqs.append(word_sa)
            seqs.append(word_sb)
        self.idf_weight = utils.idf_calculator(seqs)
        self.vocab = utils.word2index(self.idf_weight)
        self.vocab, self.embeddings = utils.load_word_embedding(self.vocab, self.emb_file)
        # self.embeddings = utils.load_embedding_from_text(self.emb_file)


    def extract(self, train_instance):

        word_sa, word_sb = train_instance.get_word(type=self.word_type,
                                                   stopwords=self.stopwords,
                                                   lower=self.lower)
        pooling_vec_sa = minavgmaxpooling(word_sa, self.vocab, self.embeddings, self.dim,
                                          convey='idf', idf_weight=self.idf_weight)
        pooling_vec_sb = minavgmaxpooling(word_sb, self.vocab, self.embeddings, self.dim,
                                          convey='idf', idf_weight=self.idf_weight)
        all_feats, all_names = vk.get_all_kernel(pooling_vec_sa, pooling_vec_sb)
        features = all_feats

        infos = [self.emb_name, self.word_type, self.stopwords, self.lower]
        return features, infos