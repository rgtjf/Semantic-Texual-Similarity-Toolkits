# coding: utf8
from __future__ import print_function

from features_dependency import *
from features_embedding import *
from features_ngram import *
from features_nn import *

from features_sequence import *

# from features_pos import *
# from features_ner import *
# from features_remove_nonlinear_kernel import *
# from evaluation import *
# from record import *
from stst.model import Model

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
    en_model_min = Model('S1-min', min)
    en_model_lr = Model('S1-lr', sklearn_lr)
    en_model_mlp = Model('S1-mlp', sklearn_mlp)

    model_sk_svr = Model('S1-svr', sklearn_svr)
    model_gb = Model('S2-gb', sklearn_gb)
    model_svr = Model('S1-liblinearsvr', liblinearsvr)
    model_rf = Model('S1-rf', rf)
    model_xgb = Model('S1-xgb', xgb)


    model = model_gb
    en_model = en_model_avg


    """
    it only make features offline, which means you can only add(load=True) or add() when train the model
    """


    file_list = get_wmt_file_list()
    print('\n'.join(file_list))
    sentences, sentence_tags = get_all_instance(file_list)
    print(sentences[:5])
    print(sentence_tags[:5])

    idf_dict = utils.IDFCalculator(sentences)


    tagged_sentences = []
    for sentence, tag in zip(sentences, sentence_tags):
        tagged_sentences.append(TaggedDocument(words=sentence, tags=[tag]))

    doc2vec_model = Doc2Vec(tagged_sentences, size=25, window=3, min_count=0, workers=10, iter=1000)

    doc2vec_model.save(config.EX_DICT_DIR + '/doc2vec.model')

    # model.add(Doc2VecGlobalFeature())

    # model.make_feature_file()
