# coding: utf8
"""
1. 每个文档自己训练一个Doc2Vec
2. 所有文档一起训练一个Doc2Vec
"""
import json

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from stst.modules.features import Feature
from stst import utils
from stst import config
from stst.data import dict_utils
from stst.libs.kernel import vector_kernel as vk


class Doc2VecFeature(Feature):

    def extract_instances(self, train_instances):
        sentences = []
        for idx, train_instance in enumerate(train_instances):
            sa, sb = train_instance.get_word(type='lemma', lower=True)
            sentences.append(TaggedDocument(words=sa, tags=['sa_%d' % idx]))
            sentences.append(TaggedDocument(words=sb, tags=['sb_%d' % idx]))

        model = Doc2Vec(sentences, size=25, window=3, min_count=0, workers=10, iter=1000)

        features = []
        infos = []
        for idx in range(len(train_instances)):
            vec_a = model.docvecs['sa_%d' % idx]
            vec_b = model.docvecs['sb_%d' % idx]
            feature, info = vk.get_all_kernel(vec_a, vec_b)
            features.append(feature)
            infos.append([])
            # infos.append([vec_a, vec_b])

        return features, infos

    # def load_instances(self, train_instances):
    #     """
    #     extract cosine distance from already trained feature file
    #     without modify the feature_file
    #     this function's priority is higher that the above extract_instances
    #     """
    #
    #     _features, _n_dim, _n_instance = Feature.load_feature_from_file(self.feature_file)
    #     features = []
    #     infos = []
    #     ''' get features from train instances'''
    #     for _feature in _features:
    #         feature = Feature._feat_string_to_list(_feature, _n_dim)
    #         features.append([feature[1]])
    #         infos.append(['cosine'])
    #
    #     features = [ Feature._feat_list_to_string(feature) for feature in features ]
    #
    #     return features, 1, _n_instance


class Doc2VecGlobalFeature(Feature):

    def __init__(self, **kwargs):
        super(Doc2VecGlobalFeature, self).__init__(**kwargs)

    def extract_instances(self, train_instances):
        model = dict_utils.DictLoader().load_doc2vec()
        file_name  = self.train_file.split('/')[-1]
        features = []
        infos = []
        for idx in range(len(train_instances)):
            vec_a = model.docvecs['%s_%d_sa' % (file_name, idx)]
            vec_b = model.docvecs['%s_%d_sb' % (file_name, idx)]
            # train_instance = train_instances[idx]
            # sa, sb = train_instance.get_word(type='lemma', stopwords=True, lower=True)
            # vec_a = model.infer_vector(sa)
            # vec_b = model.infer_vector(sb)

            feature, info = vk.get_all_kernel(vec_a, vec_b)
            features.append(feature)
            infos.append(info)

        return features, infos


class ICLRScoreFeature(Feature):
    def __init__(self, nntype, **kwargs):
        super(ICLRScoreFeature, self).__init__(**kwargs)
        self.nntype = nntype
        self.feature_name = self.feature_name + '-%s' % (nntype)

    def extract_instances(self, train_instances):
        features = []
        infos = []

        input_file = self.feature_file.split('/')[-2] + '.txt'
        f_in = utils.create_read_file(config.NN_FEATURE_PATH + '/' + self.nntype + '/' + input_file)
        for line in f_in:
            line = line.strip()
            obj = json.loads(line)
            sc = obj[0] / 5.0
            features.append([sc])
            infos.append([])

        print(len(features), features[0])

        return features, infos


class ICLRVectorFeature(Feature):
    def __init__(self, nntype, **kwargs):
        super(ICLRVectorFeature, self).__init__(**kwargs)
        self.nntype = nntype
        self.feature_name = self.feature_name + '-%s' % (nntype)

    def extract_instances(self, train_instances):
        features = []
        infos = []
        input_file = self.feature_file.split('/')[-2] + '.txt'
        f_in = utils.create_read_file(config.NN_FEATURE_PATH + '/' + self.nntype + '/' + input_file)
        for line in f_in:
            line = line.strip()
            obj = json.loads(line)
            emb1 = obj[1]
            emb2 = obj[2]
            emb1 = vk.normalize(emb1)
            emb2 = vk.normalize(emb2)
            feats, info = vk.get_all_kernel(emb1, emb2)
            features.append(feats)
            infos.append(info)

        print(len(features), features[0], infos[0])

        return features, infos
