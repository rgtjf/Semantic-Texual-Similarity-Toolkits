# coding: utf8
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import config
from stst.data_tools import data_utils

train_file = config.TRAIN_FILE
train_gs = config.TRAIN_GS_FILE
train_parse_data = data_utils.load_parse_data(train_file, train_gs, flag=False)

sa=  [u'A', u'cat', u'standing', u'on', u'tree', u'branches', u'.']
sb = [u'Two', u'green', u'and', u'white', u'trains', u'sitting', u'on', u'the', u'tracks', u'.']
sentences = []
for idx, train_instance in enumerate(train_parse_data):
    sa, sb = train_instance.get_word(type='word')
    sentences.append(TaggedDocument(words=sa, tags=['sa_%d'%idx]))
    sentences.append(TaggedDocument(words=sb, tags=['sb_%d'%idx]))

# print(sentences[:2])
model = Doc2Vec(sentences, size=50, window=5, min_count=0, workers=6, iter=10)
#
# model.build_vocab(sentences)
# model.train(sentences)
#
# print(model.docvecs[0])
# print(model.docvecs['sa_0'])

sys = []
for idx in range(len(train_parse_data)):
    sys.append(model.docvecs.similarity('sa_%d'%idx, 'sb_%d'%idx))
    #print(model.docvecs.similarity('sa_%d'%idx, 'sb_%d'%idx))
gs = open(train_gs).readlines()
gs = [ eval(g) for g in gs ]
import scipy.stats as meas
print(meas.pearsonr(sys, gs)[0])



