# coding: utf8
from __future__ import print_function

from stst.features.features_mt import *
from main_tools import *
from stst.model import Model


def NER(train_instances):
    for train_instance in train_instances:
        word_sa, word_sb = train_instance.get_word(type='lemma', lower=True)
        ner_sa, ner_sb = train_instance.get_word(type='ner')

        word_sa = [ (word, ner) for word, ner in zip(word_sa, ner_sa) if ner != 'O']
        word_sb = [ (word, ner) for word, ner in zip(word_sb, ner_sb) if ner != 'O']
        print(word_sa, word_sb)

if __name__ == '__main__':
    train_file = config.TRAIN_FILE
    train_gs = config.TRAIN_GS_FILE
    train_parse_data = data_utils.load_parse_data(train_file, train_gs, flag=False)

    rf50 = Classifier(RandomForest(n_estimators=50))
    rf = Classifier(RandomForest())
    # xgb = Classifier(XGBOOST())
    sklearn_svr = Classifier(sklearn_SVR())
    sklearn_lr = Classifier(sklearn_LinearRegression())
    sklearn_gb = Classifier(sklearn_GB())
    avg = Classifier(AvgEnsembel())
    # en_model = Model('en-lr', sklearn_lr)
    # model = Model('S1-svr', sklearn_svr)
    model = Model('es2-gb', sklearn_gb)
    model1 = Model('S1-rf', rf)
    # model2 = Model('S1-xgb', xgb)
    load = True

    ''' NER Feature - little effect'''
    # model.add(NERMatchFeature(load=False))
    # model.add(NERWordFeature(load=False))

    ''' MT Feature '''
    model.add(AsiyaEsEsMTFeature())
    # train_en(model)

    # test_en_es(model, translator='googleapi')
    # test_en_es(model, translator='googleapi_v3')

    # predict_en(model, dev_flag=False)
    # predict_snli(model, dev_flag=False)
    predict_wmt(model, translator='googleapi_v3', dev_flag=False)

    # model.make_feature_file(train_parse_data, train_file)
    # train_en(model)
    # test_en(model)
    # predict_en(model, dev_flag=False)
