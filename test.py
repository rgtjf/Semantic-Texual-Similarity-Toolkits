import stst
from main_tools import *

# Define Model
gb = stst.Classifier(stst.GradientBoostingRegression())
model = stst.Model('S1-gb', gb)

# Add features to the Model
model.add(stst.AsiyaMTFeature())

model.add(stst.SequenceFeature())
model.add(stst.SentenceFeature())
model.add(stst.ShortSentenceFeature())

model.add(stst.nGramOverlapFeature(type='lemma'))
model.add(stst.nGramOverlapFeature(type='word'))
model.add(stst.nCharGramOverlapFeature(stopwords=True))
model.add(stst.nCharGramOverlapFeature(stopwords=False))

model.add(stst.nGramOverlapBeforeStopwordsFeature(type='lemma'))
model.add(stst.nGramOverlapBeforeStopwordsFeature(type='word'))

model.add(stst.WeightednGramMatchFeature(type='lemma'))
model.add(stst.WeightednGramMatchFeature(type='word'))

model.add(stst.BOWFeature(stopwords=False))
# model.add(stst.BOWFeature(stopwords=True))
# model.add(stst.BOWGlobalFeature(stopwords=False))
# model.add(stst.BOWGlobalFeature(stopwords=True))

model.add(stst.DependencyGramFeature(convey='count'))
model.add(stst.DependencyGramFeature(convey='idf'))
model.add(stst.DependencyRelationFeature(convey='count'))
model.add(stst.DependencyRelationFeature(convey='idf'))

model.add(stst.AlignmentFeature())
model.add(stst.IdfAlignmentFeature())
model.add(stst.PosAlignmentFeature())

word2vec_file = '/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin'
paragram_file = '/home/junfeng/paragram-embedding/paragram_300_sl999.txt'
glove100_file = '/home/junfeng/GloVe/glove.6B.100d.txt'
glove300_file = '/home/junfeng/GloVe/glove.840B.300d.txt'

model.add(stst.MinAvgMaxEmbeddingFeature('word2vec', 300, word2vec_file, binary=True))
model.add(stst.MinAvgMaxEmbeddingFeature('paragram', 300, paragram_file))
model.add(stst.MinAvgMaxEmbeddingFeature('glove100', 100, glove100_file))
model.add(stst.MinAvgMaxEmbeddingFeature('glove300', 300, glove300_file))

model.add(stst.POSLemmaMatchFeature(stopwords=True))
model.add(stst.POSLemmaMatchFeature(stopwords=False))
model.add(stst.POSNounEmbeddingFeature('word2vec', 300, word2vec_file, binary=True))
model.add(stst.POSNounEditFeature())
model.add(stst.POSTreeKernelFeature())

model.add(stst.Doc2VecGlobalFeature())
model.add(stst.NegativeFeature())


hill_climbing(model)


# train and test
train_file = './data/stsbenchmark/sts-train.csv'
dev_file  = './data/stsbenchmark/sts-dev.csv'
test_file = './data/stsbenchmark/sts-test.csv'

# init the server and input the address
nlp = stst.StanfordNLP('http://localhost:9000')

# parse data
train_instances = stst.load_parse_data(train_file, nlp)
dev_instances = stst.load_parse_data(dev_file, nlp)

# train and test
model.train(train_instances, train_file)
model.test(dev_instances, dev_file)

# evaluation
dev_pearsonr = stst.eval_output_file(model.output_file)
print('Dev:', dev_pearsonr)

# test on new data set
test_instances= stst.load_parse_data(test_file, nlp)
model.test(test_instances, test_file)
test_pearsonr = stst.eval_output_file(model.output_file)
print('Test:', test_pearsonr)

recod_file = './data/records.csv'
stst.record(recod_file, dev_pearsonr, test_pearsonr, model)
