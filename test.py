import stst
import config
from stst.features.features_sequence import SequenceFeature, SentenceFeature
from stst.features.features_ngram import nGramOverlapFeature, nCharGramOverlapFeature
from stst.features.features_bow import BOWFeature, BOWCountFeature
from stst.features.features_embedding import MinAvgMaxEmbeddingFeature
from stst.evaluation import Evaluation, AdvancedEvaluation

# Define Model
gb = stst.Classifier(stst.AverageEnsemble())
model = stst.Model('U', gb)

# Add features to the Model
model.add(nGramOverlapFeature(load=True))
# model.add(nCharGramOverlapFeature(load=False))

# model.add(SequenceFeature(load=False))

model.add(BOWFeature(load=True))
# model.add(BOWCountFeature())

fastext_file = '../data/wiki.zh+.100.vec'
cco_emb_file = '/disk2/junfeng.tjf/workSpace/insuranceQA-cnn-lstm/data/word2vec/kg_ry_single_column_text_kg_ry_word2vec_vector_comment.tsv'

# model.add(MinAvgMaxEmbeddingFeature('fastext', 100, fastext_file))
# model.add(MinAvgMaxEmbeddingFeature('cco_emb', 100, cco_emb_file))


# train and test
test_file = config.TRAIN_FILE

# parse data
test_instances = stst.load_parse_data(test_file, flag=False)

# train and test
model.test(test_instances, test_file)

# evaluation
e = Evaluation(model.output_file)
print('Test:', e.eval_P_at_N(), e.eval_MAP(), e.eval_MRR())

e.plot()

ae = AdvancedEvaluation(model.output_file).case_study()

# recod_file = './data/records.csv'
# stst.record(recod_file, dev_pearsonr, test_pearsonr, model)
