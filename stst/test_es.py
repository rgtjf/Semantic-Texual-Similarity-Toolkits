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

from classifier import *
from main_tools import *
from model import Model




if __name__ == '__main__':
    rf50 = Classifier(RandomForest(n_estimators=50))
    rf = Classifier(RandomForest())
    xgb = Classifier(XGBOOST())
    liblinearsvr = Classifier(LIB_LINEAR_SVR())
    sklearn_svr = Classifier(sklearn_SVR())
    sklearn_lr = Classifier(sklearn_LinearRegression())
    sklearn_gb = Classifier(sklearn_GB())
    sklearn_mlp = Classifier(sklearn_MLP())

    avg = Classifier(AvgEnsembel())

    min = Classifier(MinEnsembel())

    en_model_avg = Model('es-avg', avg)
    model_avg = Model('es-mt-avg', avg)

    en_model_min = Model('es-min', min)
    en_model_lr = Model('es-lr', sklearn_lr)
    en_model_mlp = Model('es-mlp', sklearn_mlp)

    model_sk_svr = Model('es-svr', sklearn_svr)
    model_svr = Model('es-liblinearsvr', liblinearsvr)
    model_gb = Model('es-gb', sklearn_gb)
    model_rf = Model('es-rf', rf)
    model_xgb = Model('es-xgb', xgb)


    model = model_gb
    en_model = en_model_avg


    model.add(AsiyaMTFeature())
    model.add(SequenceFeature())
    model.add(BOWFeature(stopwords=False))
    # model.add(Doc2VecGlobalFeature())

    model.add(AlignmentFeature())
    # model.add(IdfAlignmentFeature())
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

    model.add(MinAvgMaxEmbeddingFeature(emb_type='word2vec', dim=300))
    model.add(MinAvgMaxEmbeddingFeature(emb_type='paragram', dim=300))
    model.add(MinAvgMaxEmbeddingFeature(emb_type='glove', dim=100))
    model.add(MinAvgMaxEmbeddingFeature(emb_type='glove300', dim=300))

    # train_en(model)

    model_xgb.feature_list = model.feature_list
    model_rf.feature_list = model.feature_list

    # # train_en(model_rf)
    # # train_en(model_xgb)
    # #
    # # test_es(model_rf)
    # # test_es(model_xgb)
    #
    # en_model.add(model_rf)
    # en_model.add(model_xgb)
    en_model.add(model)
    #
    # en_model.add(ICLRScoreFeature(nntype='word'))
    # en_model.add(ICLRScoreFeature(nntype='proj'))
    # en_model.add(ICLRScoreFeature(nntype='lstm'))
    # en_model.add(ICLRScoreFeature(nntype='dan'))
    #
    # en_model.add(IdfAlignmentFeature())
    #
    #
    # en_model.add(EnNegativeFeature(penalty=-0.4))

    # en_model.add(AsiyaMTFeature())

    # en_model.add(AsiyaEsEsMTFeature())

    model_avg.add(AsiyaEsEsMTFeature())
    en_model.add(model_avg)
    en_model.add(IdfAlignmentFeature())
    # en_model.add(model_avg)

    # test_wmt(en_model, translator='googleapi_v2', dev_flag=False)
    cv_test_wmt(model)
    cv_test_wmt(en_model)

    # hill_climbing(model, test_wmt)
    # test_en_es(model, translator='googleapi_v2', dev_flag=False)
    # test_en_es(model_rf, translator='googleapi_v2', dev_flag=False)
    # test_en_es(model_xgb, translator='googleapi_v2', dev_flag=False)
    # test_en_es(en_model, translator='googleapi_v2')

    localtime = time.asctime(time.localtime(time.time()))
    print("finish time:", localtime)