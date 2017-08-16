import stst
import config
from stst.features.features_sequence import SequenceFeature, SentenceFeature
from stst.features.features_ngram import nGramOverlapFeature, nCharGramOverlapFeature
from stst.features.features_bow import BOWFeature, BOWCountFeature
from stst.features.features_dssm import DSSMFeature, LSTMFeature
from stst.features.features_pinyin import PinYinFeature
from stst.features.features_embedding import MinAvgMaxEmbeddingFeature
from stst.features.features_w2v import Word2VecFeature
from stst.evaluation import Evaluation, AdvancedEvaluation

import yunqi_eval

# Define Model
avg = stst.Classifier(stst.AverageEnsemble())
model = stst.Model('U', avg)

model_en = stst.Model('EN', avg)

# Add features to the Model

''' Word '''
model.add(nGramOverlapFeature(type='word+', load=False))
model.add(BOWFeature(type='word+', load=False))

''' NER '''
model.add(BOWCountFeature(type='ner', load=True))

''' Char '''
model.add(PinYinFeature(load=True))
model.add(nCharGramOverlapFeature(load=True))
model.add(BOWFeature(type='char', load=True))

''' DL '''
model.add(DSSMFeature(load=True))
model.add(LSTMFeature(load=True))
# model.add(DSSMFeature(load=True))
# model.add(LSTMFeature(load=True))

''' Word2Vec '''
model.add(Word2VecFeature(load=True))

# model.add(SequenceFeature(load=False))
# model.add(BOWCountFeature())

fastext_file = '../data/wiki.zh+.100.vec'
cco_emb_file = '/disk2/junfeng.tjf/workSpace/insuranceQA-cnn-lstm/data/word2vec/kg_ry_single_column_text_kg_ry_word2vec_vector_comment.tsv'

# model.add(MinAvgMaxEmbeddingFeature('fastext', 100, fastext_file))
# model.add(MinAvgMaxEmbeddingFeature('cco_emb', 100, cco_emb_file))


# train and test
corpus_file = config.CORPUS_FILE
test_file = config.TRAIN_FILE

gold_file = config.GOLD_FILE


# parse data
test_instances = stst.load_parse_data(test_file, corpus_file, flag=False)

# train and test
model.test(test_instances, test_file)

# model_en.test(test_instances, test_file)

# evaluation
e = yunqi_eval.Evaluation(model.output_file, gold_file)

e.case_study_chitchat()

e.case_study_yunqi()

for th in range(0, 101, 10):
    e.eval_P(th / 100.)

e.eval_yunqi()
