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

    liblinearsvr = Classifier(LIB_LINEAR_SVR())
    sklearn_svr = Classifier(sklearn_SVR())
    model_sk_svr = Model('s1-svr', sklearn_svr)

    avg = Classifier(AvgEnsembel())

    en_model = Model('S6-combine2', avg)
    en_model_final = Model('S6-combine2-mt', avg)

    model_gb = Model('s1-gb', sklearn_gb)
    model_rf = Model('S4-rf', rf)
    model_xgb = Model('s1-xgb', xgb)
    model = model_rf

    model.add(nLemmaGramOverlapFeature())
    model.add(nWordGramOverlapFeature())
    model.add(nCharGramOverlapFeature())
    model.add(nCharGramNonStopwordsOverlapFeature())
    model.add(nWordGramOverlapBeforeStopwordsFeature())
    model.add(nLemmaGramOverlapBeforeStopwordsFeature())
    model.add(nLemmaGramOverlapMatchFeature())
    model.add(nLemmaGramOverlapBeforeStopwordsMatchFeature())

    model.add(AsiyaEsEsMTFeature())
    model.add(AsiyaMTFeature())
    model.add(SequenceFeature())
    model.add(BOWFeature(stopwords=False))
    model.add(Doc2VecGlobalFeature())
    # model.add(Doc2VecFeature())

    model.add(AlignmentFeature())
    # model.add(IdfAlignmentFeature())  # 放在前面特征里面容易拟合这个特征
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
    en_model.add(model_rf)
    # en_model.add(model_gb)
    en_model.add(model_xgb)

    # train_en(model_gb)
    # train_en(model_rf)
    # train_en(model_xgb)

    en_model.add(ICLRScoreFeature(nntype='word'))
    en_model.add(ICLRScoreFeature(nntype='proj'))
    en_model.add(ICLRScoreFeature(nntype='lstm'))
    en_model.add(ICLRScoreFeature(nntype='dan'))

    en_model.add(IdfAlignmentFeature())
    en_model.add(EnNegativeFeature(penalty=0.6))

    # test_wmt(model_gb)
    # cv_test_wmt(model_gb)
    # cv_test_wmt(model_xgb)
    # cv_test_wmt(model_rf)

    model_avg = Model('es-mt-avg', avg)
    model_avg.add(AsiyaEsEsMTFeature())
    en_model_final.add(model_avg)
    en_model_final.add(en_model)
    # train_wmt(model_xgb)
    # # train_wmt(model_avg)
    # en_model.add(model_avg)

    # cv_test_wmt(en_model)


    # train_wmt(model_rf)
    predict_wmt(en_model_final, dev_flag=False)
