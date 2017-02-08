# coding: utf8
from __future__ import print_function

from features_align import *
from features_basic import *
from features_dependency import *
from features_embedding import *
from features_mt import *
from features_nn import *
from features_pos import *
from features_sequence import *
from features_tree_kernels import *

from stst.features.features_ngram import *
from stst.model import Model
from stst.main.main_tools import *




if __name__ == '__main__':

    sklearn_gb = Classifier(sklearn_GB())
    rf = Classifier(RandomForest())
    xgb = Classifier(XGBOOST())

    avg = Classifier(AvgEnsembel())

    en_model = Model('S3-combine', avg)

    model_gb = Model('S1-gb', sklearn_gb)
    model_rf = Model('S2-rf', rf)
    model_xgb = Model('S1-xgb', xgb)
    model = model_gb

    model.add(nLemmaGramOverlapFeature())
    model.add(nWordGramOverlapFeature())
    model.add(nCharGramOverlapFeature())
    model.add(nCharGramNonStopwordsOverlapFeature())
    model.add(nWordGramOverlapBeforeStopwordsFeature())
    model.add(nLemmaGramOverlapBeforeStopwordsFeature())
    model.add(nLemmaGramOverlapMatchFeature())
    model.add(nLemmaGramOverlapBeforeStopwordsMatchFeature())

    model.add(AsiyaMTFeature())
    model.add(SequenceFeature())
    model.add(BOWFeature(stopwords=False))
    model.add(Doc2VecGlobalFeature())

    model.add(AlignmentFeature())
    model.add(IdfAlignmentFeature())  # 放在前面特征里面容易拟合这个特征
    model.add(PosAlignmentFeature())

    model.add(DependencyRelationFeature(convey='idf'))
    model.add(DependencyRelationFeature(convey='count'))
    model.add(DependencyGramFeature(convey='idf'))
    model.add(DependencyGramFeature(convey='count'))

    model.add(NegativeFeature())
    model.add(ShortSentenceFeature())

    model.add(POSLemmaMatchFeature(stopwords=True))
    model.add(POSLemmaMatchFeature(stopwords=False))
    model.add(POSNounEmbeddingFeature(emb_type='word2vec', dim=300))
    model.add(POSNounEditFeature())
    model.add(POSTreeKernelFeature())

    model.add(MinAvgMaxEmbeddingFeature(emb_type='word2vec', dim=300, load=True))
    model.add(MinAvgMaxEmbeddingFeature(emb_type='paragram', dim=300, load=True))
    model.add(MinAvgMaxEmbeddingFeature(emb_type='glove', dim=100, load=True))
    model.add(MinAvgMaxEmbeddingFeature(emb_type='glove300', dim=300, load=True))

    model_rf.feature_list = model.feature_list
    model_xgb.feature_list = model.feature_list

    en_model.add(model_rf)
    en_model.add(model_gb)
    en_model.add(model_xgb)

    en_model.add(ICLRScoreFeature(nntype='word'))
    en_model.add(ICLRScoreFeature(nntype='proj'))
    en_model.add(ICLRScoreFeature(nntype='lstm'))
    en_model.add(ICLRScoreFeature(nntype='dan'))

    en_model.add(IdfAlignmentFeature())
    en_model.add(EnNegativeFeature(penalty=0.6))

    test_en(en_model)
    predict_en_sample(en_model)

    predict_en(en_model, dev_flag=False)
    predict_snli(en_model, dev_flag=False)
    predict_wmt(en_model, dev_flag=False)
