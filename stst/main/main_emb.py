# coding: utf8
from __future__ import print_function

from stst.data_tools import data_utils
from stst.evaluation import *
from stst.main import record
from stst.model import Model

def get_sim_model(dev_parse_data, dev_file, dev_gs):
    pass


def get_ensemble_model(dev_parse_data, dev_file, dev_gs):
    ''' Define a Model'''

    ''' classifier '''
    sklearn_LR = Classifier(sklearn_LinearRegression())
    # sklearn_Lasso = Classifier(sklearn_Lasso())
    classifier = Classifier(RandomForest())
    randomforest = Classifier(RandomForest(50))
    xgboost = Classifier(XGBOOST())

    linear_svr = Classifier(LIB_LINEAR_SVR())
    svr = Classifier(LIB_SVM_SVR())
    sklearn_svr = Classifier(sklearn_SVR())
    sklearn_gb = Classifier(sklearn_GB())
    # model = Model('S1-emb-all', classifier)

    model_ensemble = Model('S1-emb-ensemble', sklearn_svr)

    model_sim_xgb = Model('S1-sim-xgb', xgboost)
    model_sim_xgb.add(AverageEmbeddingSimilarityFeature(load=True))
    # model_sim_xgb.train(train_parse_data, train_file)
    predicts = model_sim_xgb.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model_sim_xgb.output_file, dev_gs)
    print(model_sim_xgb.model_name, pearsonr)

    model_sim_rf = Model('S1-sim-rf', classifier)
    model_sim_rf.add(AverageEmbeddingSimilarityFeature(load=True))
    # model_sim_rf.train(train_parse_data, train_file)
    predicts = model_sim_rf.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model_sim_rf.output_file, dev_gs)
    print(model_sim_rf.model_name, pearsonr)

    model1 = Model('S1-emb-w2v-gb', sklearn_gb)
    model1.add(MinMaxAvgW2VEmbeddingFeature(load=True))
    # model1.train(train_parse_data, train_file)
    predicts = model1.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model1.output_file, dev_gs)
    print(model1.model_name, pearsonr)

    model2 = Model('S1-emb-glove-gb', sklearn_gb)
    model2.add(MinMaxAvgGloVeEmbeddingFeature(load=True))
    # model2.train(train_parse_data, train_file)
    predicts = model2.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model2.output_file, dev_gs)
    print(model2.model_name, pearsonr)

    model3 = Model('S1-emb-w2v-xgb', xgboost)
    model3.add(MinMaxAvgW2VEmbeddingFeature(load=True))
    # model3.train(train_parse_data, train_file)
    predicts = model3.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model3.output_file, dev_gs)
    print(model3.model_name, pearsonr)

    model4 = Model('S1-emb-glove-xgb', xgboost)
    model4.add(MinMaxAvgGloVeEmbeddingFeature(load=True))
    # model4.train(train_parse_data, train_file)
    predicts = model4.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model4.output_file, dev_gs)
    print(model4.model_name, pearsonr)

    model5 = Model('S1-emb-w2v-rf', randomforest)
    model5.add(MinMaxAvgW2VEmbeddingFeature(load=True))
    # model5.train(train_parse_data, train_file)
    predicts = model5.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model5.output_file, dev_gs)
    print(model5.model_name, pearsonr)

    model6 = Model('S1-emb-glove-rf', randomforest)
    model6.add(MinMaxAvgGloVeEmbeddingFeature(load=True))
    # model6.train(train_parse_data, train_file)
    predicts = model6.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model6.output_file, dev_gs)
    print(model6.model_name, pearsonr)

    model_ensemble.add(model_sim_xgb)
    model_ensemble.add(model_sim_rf)
    # model_ensemble.add(model1)
    # model_ensemble.add(model2)
    # model_ensemble.add(model3)
    # model_ensemble.add(model4)
    # model_ensemble.add(model5)
    # model_ensemble.add(model6)

    model_ensemble.train(train_parse_data, train_file)
    predicts = model_ensemble.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model_ensemble.output_file, dev_gs)
    print(model_ensemble.model_name, pearsonr)


if __name__ == '__main__':
    ''' Load Data'''
    train_file = config.TRAIN_FILE
    train_gs = config.TRAIN_GS_FILE
    dev_file = config.DEV_FILE
    dev_gs = config.DEV_GS_FILE

    ''' Build SentPair Object'''
    train_parse_data, dev_parse_data = data_utils.load_parse_data(train_file, train_gs, flag=False), \
                                       data_utils.load_parse_data(dev_file, dev_gs, flag=False)

    import stst.dict_utils

    stst.dict_utils.DictCreater().create_idf(train_parse_data + dev_parse_data)

    ''' Define a Model'''


    ''' classifier '''
    sklearn_LR = Classifier(sklearn_LinearRegression())
    sklearn_Lasso = Classifier(sklearn_Lasso())
    classifier = Classifier(RandomForest())
    randomforest = Classifier(RandomForest(50))
    xgboost = Classifier(XGBOOST())

    linear_svr = Classifier(LIB_LINEAR_SVR())
    svr = Classifier(LIB_SVM_SVR())
    sklearn_svr = Classifier(sklearn_SVR())
    sklearn_gb = Classifier(sklearn_GB())
    # model = Model('S1-emb-all', classifier)

    model_ensemble = Model('S1-emb-ensemble', sklearn_LR)

    model_sim_xgb = Model('S1-sim-xgb', xgboost)
    model_sim_xgb.add(AverageEmbeddingSimilarityFeature(load=True))
    # model_sim_xgb.train(train_parse_data, train_file)
    predicts = model_sim_xgb.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model_sim_xgb.output_file, dev_gs)
    print(model_sim_xgb.model_name, pearsonr)


    model_sim_rf = Model('S1-sim-rf', classifier)
    model_sim_rf.add(AverageEmbeddingSimilarityFeature(load=True))
    # model_sim_rf.train(train_parse_data, train_file)
    predicts = model_sim_rf.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model_sim_rf.output_file, dev_gs)
    print(model_sim_rf.model_name, pearsonr)


    model1 = Model('S1-emb-w2v-gb', sklearn_gb)
    model1.add(MinMaxAvgW2VEmbeddingFeature(load=True))
    # model1.train(train_parse_data, train_file)
    predicts = model1.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model1.output_file, dev_gs)
    print(model1.model_name, pearsonr)

    model2 = Model('S1-emb-glove-gb', sklearn_gb)
    model2.add(MinMaxAvgGloVeEmbeddingFeature(load=True))
    # model2.train(train_parse_data, train_file)
    predicts = model2.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model2.output_file, dev_gs)
    print(model2.model_name, pearsonr)

    model3 = Model('S1-emb-w2v-xgb', xgboost)
    model3.add(MinMaxAvgW2VEmbeddingFeature(load=True))
    # model3.train(train_parse_data, train_file)
    predicts = model3.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model3.output_file, dev_gs)
    print(model3.model_name, pearsonr)

    model4 = Model('S1-emb-glove-xgb', xgboost)
    model4.add(MinMaxAvgGloVeEmbeddingFeature(load=True))
    # model4.train(train_parse_data, train_file)
    predicts = model4.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model4.output_file, dev_gs)
    print(model4.model_name, pearsonr)


    model5 = Model('S1-emb-w2v-rf', randomforest)
    model5.add(MinMaxAvgW2VEmbeddingFeature(load=True))
    # model5.train(train_parse_data, train_file)
    predicts = model5.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model5.output_file, dev_gs)
    print(model5.model_name, pearsonr)

    model6 = Model('S1-emb-glove-rf', randomforest)
    model6.add(MinMaxAvgGloVeEmbeddingFeature(load=True))
    # model6.train(train_parse_data, train_file)
    predicts = model6.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model6.output_file, dev_gs)
    print(model6.model_name, pearsonr)



    model_ensemble.add(model_sim_xgb)
    model_ensemble.add(model_sim_rf)
    # model_ensemble.add(model1)
    # model_ensemble.add(model2)
    # model_ensemble.add(model3)
    # model_ensemble.add(model4)
    # model_ensemble.add(model5)
    # model_ensemble.add(model6)


    model_ensemble.train(train_parse_data, train_file)
    predicts = model_ensemble.test(dev_parse_data, dev_file)
    pearsonr = eval_file(model_ensemble.output_file, dev_gs)
    print(model_ensemble.model_name, pearsonr)




    record(train_file, dev_file, pearsonr, model_ensemble)

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


# lower True False 0.85123
# all 0.8596
