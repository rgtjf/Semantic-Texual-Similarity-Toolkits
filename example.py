# coding: utf8

import stst
import config
from stst.features.features_sequence import SequenceFeature, SentenceFeature
from stst.features.features_ngram import nGramOverlapFeature, nCharGramOverlapFeature, WeightednGramOverlapFeature
from stst.features.features_bow import BOWFeature, BOWCountFeature, BOWGlobalFeature, DependencyFeature, WeightedBOWFeature
from stst.features.features_dssm import DSSMFeature, LSTMFeature
from stst.features.features_pinyin import PinYinFeature
from stst.features.features_embedding import MinAvgMaxEmbeddingFeature
from stst.features.features_w2v import Word2VecFeature
from stst.features.features_ner import NERFeature
from stst.features.fetures_type import QuestionTypeFeature
from stst.evaluation import Evaluation, AdvancedEvaluation

import yunqi_eval

# Define Model
avg = stst.Classifier(stst.AverageEnsemble())
model = stst.Model('U', avg)

model_en = stst.Model('EN', avg)

# Add features to the Model
# model.add(BOWGlobalFeature(load=False))
# model.add(BOWFeature(type='word', load=True))
# model.add(DependencyFeature(type='word_dep', convey='idf', load=True))
# model.add(DependencyFeature(type='word_dep', convey='count', load=True))
# model.add(DependencyFeature(type='pos_dep', convey='count', load=True))
# model.add(DependencyFeature(type='pos_dep', convey='idf', load=True))
# model.add(SequenceFeature(load=True))


''' Word '''
# model.add(nGramOverlapFeature(type='word', load=True))
# model.add(BOWFeature(type='word+', load=True))
model.add(WeightednGramOverlapFeature(type='word', load=True))
model.add(WeightedBOWFeature(type='word', load=True))

''' NER '''
# model.add(BOWCountFeature(type='ner', load=True))
model.add(NERFeature(load=True))

''' Char '''
model.add(PinYinFeature(load=True))
model.add(nCharGramOverlapFeature(load=True))
model.add(BOWFeature(type='char', load=True))


''' Type '''
# model.add(QuestionTypeFeature(load=False))

''' DL '''
model.add(DSSMFeature(load=True))
model.add(LSTMFeature(name='lstm0808', file_name='lstm-08-08-scores.txt'))
model.add(LSTMFeature(name='lstmfeat', file_name='lstmfeat-08-23-scores.txt'))

''' Word2Vec '''
model.add(Word2VecFeature(load=True))



fastext_file = '../data/wiki.zh+.100.vec'
cco_emb_file = '/disk2/junfeng.tjf/workSpace/insuranceQA-cnn-lstm/data/word2vec/kg_ry_single_column_text_kg_ry_word2vec_vector_comment.tsv'

# model.add(MinAvgMaxEmbeddingFeature('fastext', 100, fastext_file))
# model.add(MinAvgMaxEmbeddingFeature('cco_emb', 100, cco_emb_file))


# train and test
corpus_file = config.CORPUS_FILE
test_file = config.TRAIN_FILE

gold_file = config.GOLD_FILE


# parse data
test_instances = stst.load_parse_data(test_file, corpus_file, flag=True)

# train and test
model.test(test_instances, test_file)

# model_en.test(test_instances, test_file)

# evaluation
e = yunqi_eval.Evaluation(model.output_file, gold_file)

e.case_study_chitchat()

e.case_study_yunqi_unk()

e.case_study_yunqi()

for th in range(0, 101, 10):
    e.eval_P(th / 100.)

e.eval_yunqi()

e.output()