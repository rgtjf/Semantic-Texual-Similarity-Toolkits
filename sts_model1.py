# coding: utf8

import stst
import sts_tools

# Define Model
gb = stst.Classifier(stst.GradientBoostingRegression())
model = stst.Model('S1-gb', gb)

# avg = stst.Classifier(stst.AverageEnsemble())
# model = stst.Model('S1-avg', avg)

# Add features to the Model

'''several nGramOverlapFeatures'''
model.add(stst.nGramOverlapFeature(type='word'))
model.add(stst.nGramOverlapFeature(type='lemma'))
model.add(stst.nCharGramOverlapFeature(stopwords=False))
model.add(stst.nCharGramOverlapFeature(stopwords=True))
model.add(stst.nGramOverlapBeforeStopwordsFeature(type='word'))
model.add(stst.nGramOverlapBeforeStopwordsFeature(type='lemma'))
model.add(stst.WeightednGramOverlapFeature(type='word'))
model.add(stst.WeightednGramOverlapFeature(type='lemma'))

'''several BOWFeatures'''
model.add(stst.BOWFeature(stopwords=False))
model.add(stst.BOWFeature(stopwords=True))

'''several AlignmentFeatures'''
model.add(stst.AlignmentFeature())
model.add(stst.IdfAlignmentFeature())
model.add(stst.PosAlignmentFeature())

'''several WordEmbeddingFeatures'''
word2vec_file = '/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin'
paragram_file = '/home/junfeng/paragram-embedding/paragram_300_sl999.txt'
glove100_file = '/home/junfeng/GloVe/glove.6B.100d.txt'
glove300_file = '/home/junfeng/GloVe/glove.840B.300d.txt'
model.add(stst.MinAvgMaxEmbeddingFeature('word2vec', 300, word2vec_file, binary=True))
model.add(stst.MinAvgMaxEmbeddingFeature('paragram', 300, paragram_file))
model.add(stst.MinAvgMaxEmbeddingFeature('glove100', 100, glove100_file))
model.add(stst.MinAvgMaxEmbeddingFeature('glove300', 300, glove300_file))

'''several DependencyFeatures'''
model.add(stst.DependencyGramFeature(convey='count'))
model.add(stst.DependencyGramFeature(convey='idf'))
model.add(stst.DependencyRelationFeature(convey='count'))
model.add(stst.DependencyRelationFeature(convey='idf'))


model.add(stst.POSLemmaMatchFeature(stopwords=True))
model.add(stst.POSLemmaMatchFeature(stopwords=False))
model.add(stst.POSNounEmbeddingFeature('word2vec', 300, word2vec_file, binary=True))
model.add(stst.POSNounEditFeature())
model.add(stst.POSTreeKernelFeature())

model.add(stst.Doc2VecGlobalFeature())

# model.add(stst.AsiyaMTFeature())

# model.add(stst.SequenceFeature())
# model.add(stst.SentenceFeature())
# model.add(stst.ShortSentenceFeature())

'''several NegativeFeature'''
# model.add(stst.NegativeFeature())

# sts_tools.feature_importance(model)
sts_tools.train_sts(model)
sts_tools.dev_sts(model)
sts_tools.test_sts(model)