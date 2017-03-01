# coding: utf8
from __future__ import print_function

from classifier import *
import config
import data_utils
import sentpair
from features_ngram import *
from features_embedding import *
from features_sequence import *
from features_dependency import *
from features_nn import *
from features_pos import *
from features_tree_kernels import *
from features_align import *
from features_mt import *
from features_basic import *
from model import Model
from main_tools import *




if __name__ == '__main__':

    sklearn_gb = Classifier(sklearn_GB())
    model_gb = Model('STS-gb', sklearn_gb)
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


    train_sts(model, flag=False)
    test_sts(model, flag=False)
    predict_sts(model, flag=False)

    # train_en(model)
    # test_en(model)
    # predict_en_sample(model)
    #
    # predict_en(model, dev_flag=False)
    # predict_snli(model, dev_flag=False)
    # predict_wmt(model, dev_flag=False)
