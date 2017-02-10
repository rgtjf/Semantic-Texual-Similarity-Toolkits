# coding: utf8
from __future__ import print_function

from stst.data_tools import data_utils
from stst.evaluation import *
from stst.features.features_ngram import *
from stst.model import Model
from stst.record import *


def make_model(model, train_parse_data, train_file, dev_parse_data, dev_file):

    model.train(train_parse_data, train_file)
    predicts = model.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model.output_file, dev_gs)

    print(model.model_name, pearsonr)

    return model

def model_S1(train_parse_data, train_file, dev_parse_data, dev_file, make_train_features=False, make_dev_features=False, train_classifier=False):
    pass





def test_all_dataset(model):
    test_files = config.TEST_FILES

    # TEST_FILE = TEST_DIR + '/sts-en-es/googleapi/STS.googleapi.news.txt'
    # TEST_GS_FILE = TEST_DIR + '/sts-en-es/STS.gs.news.txt'

    flag = 0
    for task in test_files:
        for corpus in test_files[task]:
            translators = ['googleapi', 'microsoftapi']
            if 'ar' in task:
                translators.append('manual')
            for translator in translators:
                if 'wmt' in corpus and 'googleapi' in translator:
                    continue
                dev_file = config.TEST_DIR + '/' + task + '/' + translator + '/' \
                           + corpus.replace('input', translator.replace('microsoftapi', 'msapi'))
                dev_gs_file = config.TEST_DIR + '/' + task + '/' + corpus
                dev_gs_file = dev_gs_file.replace('input', 'gs')
                if flag < 4:
                    flag += 1
                    continue
                    # dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=False)
                else:
                    dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs_file, flag=False)

                model.test(dev_parse_data, dev_file)
                pearsonr = eval_file(model.output_file, dev_gs_file)
                print(pearsonr)

                record('input.all.txt', dev_file, pearsonr, model)





if __name__ == '__main__':
    ''' Load Data'''
    train_file = config.TRAIN_FILE
    train_gs = config.TRAIN_GS_FILE
    dev_file = config.DEV_FILE
    dev_gs = config.DEV_GS_FILE

    ''' Build SentPair Object'''
    train_parse_data  = data_utils.load_parse_data(train_file, train_gs, flag=False)
    dev_parse_data = data_utils.load_parse_data(dev_file, dev_gs, flag=False)
    ''' Build Dict Object '''
    # create Vocab and calculate idf dict
    import stst.dict_utils

    stst.dict_utils.DictCreater().create_idf(train_parse_data)

    ''' Define a Model'''

    ''' classifier '''
    classifier = Classifier(RandomForest())
    sklearn_svr = Classifier(sklearn_SVR())

    model = Model('S1-ngram-match', sklearn_svr)

    # model.add(nLemmaGramOverlapFeature(load=False))
    # model.add(nWordGramOverlapFeature(load=False))
    # model.add(nCharGramOverlapFeature(load=False))

    # model.add(nCharGramNonStopwordsOverlapFeature(load=True))
    # model.add(nWordGramOverlapBeforeStopwordsFeature(load=True))
    # model.add(nLemmaGramOverlapBeforeStopwordsFeature(load=True))

    model.add(nCharGramNonStopwordsOverlapMatchFeature(load=False))
    model.add(nWordGramOverlapBeforeStopwordsMatchFeature(load=False))
    model.add(nLemmaGramOverlapBeforeStopwordsMatchFeature(load=False))



    ''' alignment feature '''
    # model.add(AlignmentFeature(load=True))

    ''' wordnet feature '''
    # model.add(WordNetFeatures(load=True))

    # model2 = Model('S1-all-others', classifier)
    #
    # ''' location feature '''
    # model2.add(LocationFeature(load=True))
    #
    # ''' length feature '''
    # model2.add(LengthFeature(load=True))
    #
    # ''' pos feature '''
    # model2.add(POSFeature(load=True))
    #
    # make_model(model2, train_parse_data, train_file, dev_parse_data, dev_file)
    #
    # ''' embeddeing feature '''
    # from main_emb import get_ensemble_model
    #
    # randomforest = Classifier(RandomForest(50))
    # xgboost = Classifier(XGBOOST())
    #
    # linear_svr = Classifier(LIB_LINEAR_SVR())
    # svr = Classifier(LIB_SVM_SVR())
    # sklearn_svr = Classifier(sklearn_SVR())
    # sklearn_gb = Classifier(sklearn_GB())
    # # model = Model('S1-emb-all', classifier)
    #
    # #model_ensemble = Model('S1-emb-ensemble', sklearn_svr)
    #
    # model_sim_xgb = Model('S1-sim-xgb', xgboost)
    # model_sim_xgb.add(AverageEmbeddingSimilarityFeature(load=True))
    # # model_sim_xgb.train(train_parse_data, train_file)
    # predicts = model_sim_xgb.test(dev_parse_data, dev_file)
    # pearsonr = eval_file(model_sim_xgb.output_file, dev_gs)
    # print(model_sim_xgb.model_name, pearsonr)
    #
    # model_sim_rf = Model('S1-sim-rf', classifier)
    # model_sim_rf.add(AverageEmbeddingSimilarityFeature(load=True))
    # # model_sim_rf.train(train_parse_data, train_file)
    # predicts = model_sim_rf.test(dev_parse_data, dev_file)
    # pearsonr = eval_file(model_sim_rf.output_file, dev_gs)
    # print(model_sim_rf.model_name, pearsonr)

    # model_emb = get_ensemble_model(dev_parse_data, dev_file, dev_gs)

    # model.add(model2)
    # model.add(model_sim_xgb)
    # model.add(model_sim_rf)

    model.train(train_parse_data, train_file)
    predicts = model.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model.output_file, dev_gs)
    print(pearsonr)

    record(train_file, dev_file, pearsonr, model)

    test_all_dataset(model)

    # model.add(TriGramFeature('trigram'))
    #
    # emb_model = Model('name', classifier)
    # emb_model.add(Fearures)
    # emb_model.add(Features)
    #
    # model.add(emb_model)
    #
    # model.train()
    #
    # model.test()
    # model.predict() # ?
