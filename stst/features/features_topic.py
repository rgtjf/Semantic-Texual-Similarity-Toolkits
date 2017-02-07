# coding=utf-8
import dict_utils
from features import Feature

class TopicFeature(Feature):

    def extract(self, train_instance):
        topic = dict_utils.DictLoader().load_dict('topic')
        sa, sb = train_instance.get_word(type='word')
        sa = [ topic.get(w, 0) for w in sa ]
        sb = [ topic.get(w, 0) for w in sb]

        features, infos = [], []
        return features, infos