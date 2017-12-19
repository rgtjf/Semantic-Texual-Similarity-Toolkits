# coding: utf8
from __future__ import print_function

from stst.modules.features import Feature
from stst import utils
from stst.libs.kernel import vector_kernel as vk


class POSMatchFeature(Feature):
    def extract_information(self, train_instances):
        pass

    def extract(self, train_instance):
        pos_sa, pos_sb = train_instance.get_word(type='pos', stopwords=True)
        features, infos = utils.sentence_match_features(pos_sa, pos_sb)
        return features, infos


class POSVectorFeature(Feature):
    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            pos_sa, pos_sb = train_instance.get_word(type='pos', stopwords=True)
            seqs.append(pos_sa)
            seqs.append(pos_sb)
        self.idf_weight = utils.idf_calculator(seqs)  # idf weight is different

    def extract(self, train_instance):
        pos_sa, pos_sb = train_instance.get_word(type='pos', stopwords=True)
        features, infos = utils.sentence_vectorize_features(pos_sa, pos_sb, self.idf_weight, convey='count')
        return features, infos


class POSFeature(Feature):
    def extract(self, train_instance):
        pos_sa, pos_sb = train_instance.get_pos_tag()
        # noun_sa: [ [I, NN], [love, VP], [Shanghai, NNJ] ]
        features = []
        infos = []
        for tag in ['n', 'v', 'a', 'r', '#']:
            num_sa = num_sb = 0
            for lemma, pos in pos_sa:
                if pos == tag:
                    num_sa += 1
            for lemma, pos in pos_sb:
                if pos == tag:
                    num_sb += 1
            p, r, f1 = 0.0, 0.0, 1.0
            if num_sa > 0 and num_sb > 0:
                p = 1.0 * num_sa / num_sb
                r = 1.0 * num_sb / num_sa
                f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
            features += [f1, num_sa + num_sb]
            infos.append((tag, num_sa, num_sb))
        return features, infos


class POSLemmaMatchFeature(Feature):
    def __init__(self, stopwords, **kwargs):
        super(POSLemmaMatchFeature, self).__init__(**kwargs)
        self.stopwords = stopwords
        self.feature_name = self.feature_name + '-%s' % (stopwords)

    def extract(self, train_instance):
        pos_sa, pos_sb = train_instance.get_pos_tag(stopwords=self.stopwords)
        # noun_sa: [ [I, NN], [love, VP], [Shanghai, NNJ] ]
        features = []
        infos = []
        for tag in ['n', 'v', 'a', 'r', '#']:
            sa = [ w for w, ner in pos_sa if ner == tag]
            sb = [ w for w, ner in pos_sb if ner == tag]

            feats = [utils.jaccrad_coeff(sa, sb), utils.dice_coeff(sa, sb)]
            # feats, info = utils.sequence_match_features(sa, sb)
            features += feats
            info = [sa, sb]
            infos.append(info)

        return features, infos


class POSNounEditFeature(Feature):
    def extract(self, train_instance):
        pos_sa, pos_sb = train_instance.get_pos_tag(stopwords=False)
        # noun_sa: [ [I, NN], [love, VP], [Shanghai, NNJ] ]

        sa = [w for w, ner in pos_sa if ner == 'n']
        sb = [w for w, ner in pos_sb if ner == 'n']

        features, infos = utils.sentence_sequence_features(sa, sb)
        return features, infos


class POSNounEmbeddingFeature(Feature):
    def __init__(self, emb_name, dim, emb_file, **kwargs):
        super(POSNounEmbeddingFeature, self).__init__(**kwargs)
        self.emb_name = emb_name
        self.dim = dim
        self.emb_file = emb_file

        self.feature_name += '-%s' % (emb_name)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            pos_sa, pos_sb = train_instance.get_pos_tag(stopwords=False)
            sa = [w for w, tag in pos_sa if tag == 'n']
            sb = [w for w, tag in pos_sb if tag == 'n']
            seqs.append(sa)
            seqs.append(sb)

        idf_weight = utils.idf_calculator(seqs)
        vocab = utils.word2index(idf_weight)
        self.idf_weight = idf_weight
        self.vocab, self.embeddings = utils.load_word_embedding(vocab, self.emb_file)

    def extract(self, train_instance):
        from stst.features.features_embedding import minavgmaxpooling

        pos_sa, pos_sb = train_instance.get_pos_tag(stopwords=False)
        sa = [w for w, tag in pos_sa if tag == 'n']
        sb = [w for w, tag in pos_sb if tag == 'n']

        pooling_vec_sa = minavgmaxpooling(sa, self.vocab, self.embeddings, self.dim,
                                          convey='idf', idf_weight=self.idf_weight)
        pooling_vec_sb = minavgmaxpooling(sb, self.vocab, self.embeddings, self.dim,
                                          convey='idf', idf_weight=self.idf_weight)

        all_feats, all_names = vk.get_all_kernel(pooling_vec_sa, pooling_vec_sb)
        features = all_feats
        infos = [self.emb_name]
        return features, infos