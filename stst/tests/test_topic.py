# coding: utf8
from __future__ import print_function

from features_align import *
from features_basic import *
from features_dependency import *
from features_embedding import *
from features_mt import *
from features_ner import *
from features_ngram import *
from features_nn import *
from features_pos import *
from features_remove_nonlinear_kernel import *
from features_sequence import *
from features_tree_kernels import *

from stst.model import Model


def NER(train_instances):
    for train_instance in train_instances:
        word_sa, word_sb = train_instance.get_word(type='lemma', lower=True)
        ner_sa, ner_sb = train_instance.get_word(type='ner')

        word_sa = [(word, ner) for word, ner in zip(word_sa, ner_sa) if ner != 'O']
        word_sb = [(word, ner) for word, ner in zip(word_sb, ner_sb) if ner != 'O']
        print(word_sa, word_sb)

if __name__ == '__main__':
    train_file = config.TRAIN_FILE
    train_gs = config.TRAIN_GS_FILE
    train_parse_data = data_utils.load_parse_data(train_file, train_gs, flag=False)

    # NER(train_parse_data)

    rf50 = Classifier(RandomForest(n_estimators=50))
    rf = Classifier(RandomForest())
    xgb = Classifier(XGBOOST())
    liblinearsvr = Classifier(LIB_LINEAR_SVR())
    sklearn_svr = Classifier(sklearn_SVR())
    sklearn_lr = Classifier(sklearn_LinearRegression())
    sklearn_gb = Classifier(sklearn_GB())
    sklearn_mlp = Classifier(sklearn_MLP())

    en_avg = Classifier(AvgEnsembel())
    en_min = Classifier(MinEnsembel())

    en_model_avg = Model('S1-avg', en_avg)
    en_model_min = Model('S1-min', en_min)
    en_model_lr = Model('S1-lr', sklearn_lr)
    en_model_mlp = Model('S1-mlp', sklearn_mlp)

    model_sk_svr = Model('S1-svr', sklearn_svr)
    model_gb = Model('S1-gb', sklearn_gb)
    model_svr = Model('S1-liblinearsvr', liblinearsvr)
    model_rf = Model('S1-rf', rf)
    model_xgb = Model('S1-xgb', xgb)

    model = model_gb
    en_model = en_model_avg

    load = True

    model.add(BOWFeature(stopwords=True, load=load))
    model.add(BOWFeature(stopwords=False, load=load))

    model.add(POSLemmaMatchFeature(stopwords=True, load=load))
    model.add(POSLemmaMatchFeature(stopwords=False, load=load))
    model.add(POSNounEmbeddingFeature(emb_type='word2vec', dim=300))
    model.add(POSNounEditFeature(load=True))
    model.add(POSTreeKernelFeature(load=True))
    model.add(SequenceFeature(load=True))

    model.add(NegativeFeature())
    model.add(ShortSentenceFeature())
    model.add(SequenceFeature())

    model.add(DependencyRelationFeature(convey='idf', load=load))
    model.add(DependencyGramFeature(convey='idf', load=load))

    model.add(DependencyRelationFeature(convey='count', load=load))
    model.add(DependencyGramFeature(convey='count', load=load))

    model.add(Doc2VecFeature())

    model.add(MinAvgMaxEmbeddingFeature(emb_type='word2vec', dim=300))
    model.add(MinAvgMaxEmbeddingFeature(emb_type='paragram', dim=300))
    model.add(MinAvgMaxEmbeddingFeature(emb_type='glove', dim=100))
    model.add(MinAvgMaxEmbeddingFeature(emb_type='glove300', dim=300))

    model.add(nLemmaGramOverlapFeature())  # 3
    model.add(nWordGramOverlapFeature())  # 3 remove
    model.add(nCharGramOverlapFeature())  # 4
    model.add(nCharGramNonStopwordsOverlapFeature())  # 4
    model.add(nWordGramOverlapBeforeStopwordsFeature())  # 3 remove
    model.add(nLemmaGramOverlapBeforeStopwordsFeature())  # 3
    model.add(nLemmaGramOverlapMatchFeature())  # 3
    model.add(nLemmaGramOverlapBeforeStopwordsMatchFeature())  # 3

    model.add(AlignmentFeature())
    model.add(PosAlignmentFeature())
    model.add(IdfAlignmentFeature())

    ''' Topic Feature '''
    # model.add(TopicFeature())

    ''' NER Feature - little effect'''
    # model.add(NERMatchFeature(load=False))
    # model.add(NERWordFeature(load=False))

    ''' MT Feature '''
    model.add(AsiyaMTFeature())

    ''' ICLR Feature '''
    # model.add(ICLRScoreFeature(nntype='word'))
    # model.add(ICLRVectorFeature(nntype='word'))

    # model.add(ICLRScoreFeature(nntype='proj'))
    # model.add(ICLRVectorFeature(nntype='proj'))

    # model.add(ICLRScoreFeature(nntype='lstm'))
    # model.add(ICLRVectorFeature(nntype='lstm'))

    # model.add(ICLRScoreFeature(nntype='dan'))
    # model.add(ICLRVectorFeature(nntype='dan'))

    # model.add(DependencyRelationFeature(load=True))
    # model.add(DependencyGramFeature(load=True))
    # model.make_feature_file(train_parse_data, train_file)
    #

    model_svr.feature_list = model.feature_list
    model_rf.feature_list = model.feature_list
    model_xgb.feature_list = model.feature_list

    # train_en(model_rf)
    # train_en(model)
    # train_en(model_xgb)

    """ ensemble POS Feature to capture order information """
    pos_model = Model('pos-gb', sklearn_gb)
    pos_model.add(POSLemmaMatchFeature(stopwords=True, load=load))
    pos_model.add(POSLemmaMatchFeature(stopwords=False, load=load))
    pos_model.add(POSNounEmbeddingFeature(emb_type='word2vec', dim=300))
    pos_model.add(POSNounEditFeature(load=True))
    pos_model.add(POSTreeKernelFeature(load=True))
    pos_model.add(SequenceFeature(load=True))
    # train pos_model
    # train_en(pos_model)
    en_model.add(pos_model)

    """ ensemble model must pre_trained related models """
    en_model.add(model)
    en_model.add(model_rf)
    en_model.add(model_xgb)

    en_model.add(ICLRScoreFeature(nntype='word'))
    en_model.add(ICLRScoreFeature(nntype='proj'))
    en_model.add(ICLRScoreFeature(nntype='lstm'))
    en_model.add(ICLRScoreFeature(nntype='dan'))
    en_model.add(IdfAlignmentFeature())
    en_model.add(EnNegativeFeature(penalty=-0.6))


    # train_en(en_model)
    # test_en(en_model)

    predict_en_sample(pos_model)
    predict_en_sample(model)
    predict_en_sample(model_rf)
    predict_en_sample(model_xgb)
    predict_en_sample(en_model, dev_flag=False)

    # hill_climbing_en(model)
    localtime = time.asctime(time.localtime(time.time()))
    print("finish time:", localtime)


    # """ ensemble embedding feature """
    # emb_model = Model('emb-gb', sklearn_gb)
    # emb_model.add(MinAvgMaxEmbeddingFeature(emb_type='word2vec', dim=300))
    # emb_model.add(MinAvgMaxEmbeddingFeature(emb_type='paragram', dim=300))
    # emb_model.add(MinAvgMaxEmbeddingFeature(emb_type='glove', dim=100))
    # emb_model.add(MinAvgMaxEmbeddingFeature(emb_type='glove300', dim=300))
    # # train emb_model
    # train_en(emb_model)
    # en_model.add(emb_model)