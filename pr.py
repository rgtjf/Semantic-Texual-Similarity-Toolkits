import stst
import config
from stst.features.features_sequence import SequenceFeature, SentenceFeature
from stst.features.features_ngram import nGramOverlapFeature, nCharGramOverlapFeature
from stst.features.features_bow import BOWFeature, BOWCountFeature
from stst.features.features_embedding import MinAvgMaxEmbeddingFeature
from stst.evaluation import Evaluation, AdvancedEvaluation, AdvancedDSSMEvaluation

# train and test
score_file = config.SCORE_FILE
gold_file = config.GOLD_FILE

# evaluation
e = Evaluation(score_file, gold_file)
print('Test:', e.eval_P_at_N(), e.eval_MAP(), e.eval_MRR())

# e.plot()

# ae = AdvancedEvaluation(model.output_file).case_study()
ae = AdvancedDSSMEvaluation(score_file, gold_file).case_study()

# recod_file = './data/records.csv'
# stst.record(recod_file, dev_pearsonr, test_pearsonr, model)
