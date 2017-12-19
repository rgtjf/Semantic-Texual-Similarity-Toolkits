# coding: utf8
from stst.modules.features import Feature
from stst.libs.sentence_similarity.short_sentence_similarity import *


class ShortSentenceFeature(Feature):
    def extract(self, train_instance):
        sa, sb = train_instance.get_word(type='word')
        features = [semantic_similarity(sa, sb, True), semantic_similarity(sa, sb, False)]
        infos = []
        return features, infos