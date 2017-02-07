# coding=utf-8
"""
Ref:
    - https://rare-technologies.com/word2vec-tutorial/
    - word2vec: https://radimrehurek.com/gensim/models/word2vec.html
    - doc2vec:  https://radimrehurek.com/gensim/models/doc2vec.html
    - phrases:  https://radimrehurek.com/gensim/models/phrases.html
    - how to train: http://blog.imaou.com/opensource/2015/08/31/how_to_train_word2vec.html
    - how to use: http://blog.csdn.net/churximi/article/details/51472300
"""
from __future__ import print_function
import os, codecs
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


def train_word2vec(train_sentences, out_embedding_file=None, out_model_file=None):
    """
    hs = if 1, hierarchical softmax will be used for model training. If set to 0 (default), and negative is non-zero,
negative sampling will be used.

    negative = if > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be
drawn (usually between 5-20). Default is 5. If set to 0, no negative samping is used.

    sg = 1, Skip-gram; 0, CBOW
    """
    CPU_COUNT = 6
    WORD2VEC_LIST = [
        (0, 'CBOW_s100_w3_m0_n3_s1e3.word2vec',
         Word2Vec(sg=0, size=100, window=3, min_count=3, negative=3, sample=0.001, workers=CPU_COUNT)),
    ]
    model = WORD2VEC_LIST[0][2]

    model.build_vocab(train_sentences)
    model.train(train_sentences)

    if out_embedding_file:
        model.save_word2vec_format(out_embedding_file, binary=True)
    if out_model_file:
        model.save(out_model_file)

    return model


def kmeans_cluster(X, K):
    """
    :param X: [ emb1, emb2
                ]

    """
    kmeans = KMeans(n_clusters=K, max_iter=300, n_init=8, n_jobs=3)
    kmeans.fit(X)

    lables = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    return lables, cluster_centers


def load_data():

    ''' Build SentPair Object'''
    sentences = []

    #TODO load_data
    #return format
    # sentences = [
    #     [ 'w1', 'w2', 'w3' ],
    #     [ 'w2', 'w2', 'w3'],
    # ]

    return sentences


def embedding_kmeans():  # train_sentences, topic_number
    print('=====> load data')
    train_sentences = load_data()
    topic_number = 1024

    print('=====> train word2vec')
    model = train_word2vec(train_sentences)

    print('=====> kmeans cluster')
    X = [model[w] for w in model.vocab]
    labels, centers = kmeans_cluster(X, topic_number)

    print('=====> record file')
    record(model.vocab, labels)


def vectorize_topic(w2t_dict, sent):
    l = len(w2t_dict) + 1  # '0' for unknown word
    vec = np.zeros((l,), dtype=np.int32)
    for w in sent:
        if w in w2t_dict:
            vec[w2t_dict[w]] += 1
        else:
            vec[0] += 1
    return vec


def record(vocab, labels, file_path='topic.txt'):
    ''' vocab \t label '''
    c = sorted(zip(vocab, labels), key=lambda x: (x[1], x[0]))
    with codecs.open(file_path, 'w', encoding='utf8') as f_topic:
        for w, topic_id in c:
            print(w, topic_id + 1, file=f_topic)
            # f_topic.write('%s\t%d\n' % (w, topic_id+1))


if __name__ == '__main__':

    w2v_file = '/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin'
    model = Word2Vec.load_word2vec_format(w2v_file, binary=True)

    sentences = load_data()
    vocab = []
    for sentence in sentences:
        for word in sentence:
            vocab.append(word)

    vocab = list(set(vocab))
    vocab = [w for w in vocab if w in model.vocab]
    topic_number = 4096

    print('=====> kmeans cluster')
    X = [model[w] for w in vocab]
    labels, centers = kmeans_cluster(X, topic_number)

    print('=====> record file')
    record(vocab, labels)
