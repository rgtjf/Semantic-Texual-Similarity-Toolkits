# coding: utf8
import json

import pyprind

from stst.modules.features import Feature
from stst import utils
from stst.libs.word_aligner import aligner


class AlignmentFeature(Feature):
    def extract(self, train_instance):
        parse_sa, parse_sb = train_instance.get_parse()
        features, infos = aligner.align_feats(parse_sa, parse_sb)
        return features, infos


class IdfAlignmentFeature(Feature):
    """
        extract from features info.
        extract from one instance is not enough,
        instead, extract from the upper level i.e., extract_instaces(self, train_instances)
    """

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            lemma_sa, lemma_sb = train_instance.get_word(type='lemma', lower=True)
            seqs.append(lemma_sa)
            seqs.append(lemma_sb)

        self.idf_weight = utils.idf_calculator(seqs)

    def extract_instances(self, train_instances):
        """ extract features to features """
        self.extract_information(train_instances)

        features = []
        infos = []
        process_bar = pyprind.ProgPercent(len(train_instances))

        ''' get features from train instances'''

        alignment_feature_file = self.feature_file.replace('IdfAlignmentFeature', 'AlignmentFeature')
        alignment_features = utils.create_read_file(alignment_feature_file).readlines()

        idf_weight = self.idf_weight
        default_idf_weight = min(idf_weight.values())

        for train_instance, alignment_feature in zip(train_instances, alignment_features[1:]):
            process_bar.update()

            alignment_feature = alignment_feature.split('\t#\t')[1]
            myWordAlignments = json.loads(alignment_feature)[0]  # list of [sa_idx, sb_idx] index start from 1

            word_sa, word_sb = train_instance.get_word(type='lemma', lower=True)

            sa_aligned = [sa_idx - 1 for sa_idx, sb_idx in myWordAlignments]
            sb_aligned = [sb_idx - 1 for sa_idx, sb_idx in myWordAlignments]

            sent1_aligned = [0] * len(word_sa)
            sent2_aligned = [0] * len(word_sb)

            for sa_index in sa_aligned:
                sent1_aligned[sa_index] = 1

            for sb_index in sb_aligned:
                sent2_aligned[sb_index] = 1

            # calc all and aligned except stopwords
            sent1_sum = 0
            sent2_sum = 0
            sent1_ali = 0
            sent2_ali = 0
            for idx, word in enumerate(word_sa):
                weight = idf_weight.get(word, default_idf_weight)
                sent1_ali += sent1_aligned[idx] * weight
                sent1_sum += weight

            for idx, word in enumerate(word_sb):
                weight = idf_weight.get(word, default_idf_weight)
                sent2_ali += sent2_aligned[idx] * weight
                sent2_sum += weight
            feature = [1.0 * (sent1_ali + sent2_ali) / (sent1_sum + sent2_sum + 1e-6)]
            info = [sent2_ali, sent2_ali, sent1_sum, sent2_sum]
            features.append(feature)
            infos.append(info)

        return features, infos


class PosAlignmentFeature(Feature):
    """
    extract from features info.
    extract from one instance is not enough,
    instead, extract from the upper level i.e., extract_instaces(self, train_instances)
    """

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            lemma_sa, lemma_sb = train_instance.get_word(type='lemma', lower=True)
            seqs.append(lemma_sa)
            seqs.append(lemma_sb)

        self.idf_weight = utils.idf_calculator(seqs)

    def extract_instances(self, train_instances):
        """ extract features to features """

        self.extract_information(train_instances)
        idf_weight = self.idf_weight
        min_idf_weight = min(idf_weight.values())

        features = []
        infos = []
        process_bar = pyprind.ProgPercent(len(train_instances))

        ''' get features from train instances'''

        alignment_feature_file = self.feature_file.replace('PosAlignmentFeature', 'AlignmentFeature')
        alignment_features = utils.create_read_file(alignment_feature_file).readlines()

        for train_instance, alignment_feature in zip(train_instances, alignment_features[1:]):
            process_bar.update()

            alignment_feature = alignment_feature.split('\t#\t')[1]
            myWordAlignments = json.loads(alignment_feature)[0]  # list of [sa_idx, sb_idx] index start from 1
            pos_sa, pos_sb = train_instance.get_pos_tag(stopwords=False)
            ner_sa, ner_sb = train_instance.get_word(type='ner', stopwords=False)
            word_sa, word_sb = train_instance.get_word(type='lemma', lower=True)

            feature, info = [], []
            sa_aligned = [sa_idx - 1 for sa_idx, sb_idx in myWordAlignments]
            sb_aligned = [sb_idx - 1 for sa_idx, sb_idx in myWordAlignments]

            sent1_aligned = [0] * len(word_sa)
            sent2_aligned = [0] * len(word_sb)

            for sa_index in sa_aligned:
                sent1_aligned[sa_index] = 1

            for sb_index in sb_aligned:
                sent2_aligned[sb_index] = 1

            sent1_sum = {'n': 0., 'v': 0., 'a': 0., 'r': 0., '#': 0.}
            sent2_sum = {'n': 0., 'v': 0., 'a': 0., 'r': 0., '#': 0.}
            sent1_ali = {'n': 0., 'v': 0., 'a': 0., 'r': 0., '#': 0.}
            sent2_ali = {'n': 0., 'v': 0., 'a': 0., 'r': 0., '#': 0.}
            for idx, word in enumerate(word_sa):
                pos = pos_sa[idx][1]
                weight = idf_weight.get(word, min_idf_weight)
                sent1_ali[pos] = sent1_aligned[idx] * weight
                sent1_sum[pos] += weight

            for idx, word in enumerate(word_sb):
                pos = pos_sb[idx][1]
                weight = idf_weight.get(word, min_idf_weight)
                sent2_ali[pos] += sent2_aligned[idx] * weight
                sent2_sum[pos] += weight

            for pos in ['n', 'v', 'a', 'r', '#']:
                score = 1.0 * (sent1_ali[pos] + sent2_ali[pos]) / (sent1_sum[pos] + sent2_sum[pos] + 1e-6) \
                    if sent1_sum[pos] + sent2_sum[pos] > 1e-6 else 0.0
                feature.append(score)

            info = [sent1_sum, sent2_sum, sent1_ali, sent2_ali]

            features.append(feature)
            infos.append(info)

        return features, infos
