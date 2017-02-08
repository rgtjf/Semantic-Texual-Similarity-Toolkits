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

    en_model_avg = Model('S1-avg', avg)
    en_model_avg_2 = Model('S2-avg', avg)

    en_model_min = Model('S1-min', min)
    en_model_lr = Model('S1-lr', sklearn_lr)
    en_model_mlp = Model('S1-mlp', sklearn_mlp)

    model_sk_svr = Model('S1-svr', sklearn_svr)
    model_svr = Model('S1-liblinearsvr', liblinearsvr)
    model_gb = Model('S1-gb', sklearn_gb)
    model_rf = Model('S1-rf', rf)
    model_xgb = Model('S1-xgb', xgb)


    model = model_gb
    en_model = en_model_avg


    """
    it only make features offline, which means you can only add(load=True) or add() when train the model
    """

    load = True

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

    # model.add(ICLRScoreFeature(nntype='word'))
    # model.add(ICLRVectorFeature(nntype='word'))
    # model.add(ICLRScoreFeature(nntype='proj'))
    # model.add(ICLRVectorFeature(nntype='proj'))
    # model.add(ICLRScoreFeature(nntype='lstm'))
    # model.add(ICLRVectorFeature(nntype='lstm'))
    # model.add(ICLRScoreFeature(nntype='dan'))
    # model.add(ICLRVectorFeature(nntype='dan'))


    # ''' Topic Feature '''
    # from features_topic import TopicFeature
    # model.add(TopicFeature())
    #
    # ''' NER Feature - little effect'''
    # from features_ner import NERMatchFeature, NERWordFeature
    # model.add(NERMatchFeature(load=False))
    # model.add(NERWordFeature(load=False))
    #
    # ''' Corpus Feature(LSI) '''
    # from features_corpus import DistSim
    # model.add(DistSim())
    #
    # ''' WordNet Feature '''
    # model.add(WordNetFeatures())
    #
    # ''' Length Feature '''
    # model.add(LengthFeature())

    # train_en(model)
    # test_en(model)

    model_xgb.feature_list = model.feature_list
    model_rf.feature_list = model.feature_list

    # train_en(model_rf)
    # train_en(model_xgb)

    # test_en(model_rf)
    # test_en(model_xgb)

    # en_model.add(model_rf)
    # en_model.add(model_xgb)
    # en_model.add(model)
    #
    # en_model.add(ICLRScoreFeature(nntype='word'))
    # en_model.add(ICLRScoreFeature(nntype='proj'))
    # en_model.add(ICLRScoreFeature(nntype='lstm'))
    # en_model.add(ICLRScoreFeature(nntype='dan'))

    # en_model.add(IdfAlignmentFeature())

    # en_model.add(EnNegativeFeature(penalty=-0.1))

    # train_en(en_model)
    # test_en(en_model)
    predict_en_sample(en_model)

    # predict_en(model, dev_flag=False)
    # predict_snli(model, dev_flag=False)
    # predict_wmt(model, dev_flag=False)

    # hill_climbing_en(model)

    localtime = time.asctime(time.localtime(time.time()))
    print("finish time:", localtime)