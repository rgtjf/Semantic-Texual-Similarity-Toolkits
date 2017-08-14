# coding: utf8
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
from collections import Counter, OrderedDict

from stst.features.features import Feature
from stst import dict_utils, utils
from stst import config

class Word2VecFeature(Feature):

    def extract_information(self, train_instances):
        sents = []
        for train_instance in train_instances:
            sa, sb = train_instance.get_pair('word')
            sents.append(sa)
            sents.append(sb)
        self.idf_dict = utils.idf_calculator(sents)

        # write to file
        tmp_file = config.TMP_DIR + '/vocab.txt'
        with utils.create_write_file(tmp_file) as fw:
            sorted_idf_dict = sorted(self.idf_dict.items(), key=lambda x:x[1], reverse=False)
            for item in sorted_idf_dict:
                print('{}'.format(item[0]), file=fw)

        # os.system
        emb_file = config.TMP_DIR + '/vocab.emb.txt'
        model_file = config.WIKI_BIN_FILE
        os.system('cat {} | fasttext print-word-vectors {} > {}'.format(tmp_file, model_file, emb_file))

        # load_embfile
        self.embedding, self.vocab2id = utils.load_embedding_from_text(emb_file, 100)

    def extract(self, train_instance):
        # Warning: the length of word_sa is not the same of the length of the emb
        # If the same, how to solve the OOV UNK word's Embedding
        # Thanks to fasttext! it can generate word vector from the model
        word_sa, word_sb = train_instance.get_pair(type='word')

        def word2emb(sent, vocab, embedding):
            embs = []
            for word in sent:
                # if word in vocab:
                if word in vocab:
                    embs.append(embedding[vocab[word]])
                else:
                    embs.append(embedding[vocab['__UNK__']])
            return embs
        emb_sa = word2emb(word_sa, self.vocab2id, self.embedding)
        emb_sb = word2emb(word_sb, self.vocab2id, self.embedding)

        sent_sim = 0.0
        threshold = 0.5
        match_idxs = []
        match_words = []
        if len(emb_sa) != 0 and len(emb_sb) != 0:
            is_matched = [False] * len(emb_sb)
            for ida, emb_wa in enumerate(emb_sa):
                word_sim = 0.
                word_idx = -1
                for idb, emb_wb in enumerate(emb_sb):
                    if is_matched[idb] is True:
                        continue
                    ab_sim = 1. - utils.cosine_distance(emb_wa, emb_wb)
                    if ab_sim > word_sim:
                        word_sim = ab_sim
                        word_idx = idb
                if word_sim > threshold:
                    sent_sim += word_sim
                    is_matched[word_idx] = True
                    match_idxs.append([ida, word_idx])
                    match_words.append([word_sa[ida], word_sb[word_idx]])
        sent_sim = sent_sim / len(word_sa) if len(word_sa) != 0 else 0.0
        features = [ sent_sim ]
        infos = ['w2v', match_words]
        return features, infos
