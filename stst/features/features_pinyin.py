# coding: utf8
from __future__ import print_function
from __future__ import unicode_literals

from collections import Counter, OrderedDict

from stst.features.features import Feature
from stst import dict_utils, utils
from stst import config

class PinYinFeature(Feature):

    def extract_information(self, train_instances):
        # self.pinyin_dict = dict_utils.DictLoader().load_dict('pinyin', config.PINYIN_FILE)
        pinyin_dict = OrderedDict()
        with utils.create_read_file(config.PINYIN_FILE) as f:
            for line in f:
                items = line.strip().split()
                if items[0] not in pinyin_dict:
                    pinyin_dict[items[0]] = []
                pinyin = items[1][:-1]
                pinyin = pinyin[0].upper() + pinyin[1:]
                pinyin_dict[items[0]].append(pinyin)

        self.pinyin_dict = pinyin_dict

    def extract(self, train_instance):
        char_sa, char_sb = train_instance.get_pair(type='char')

        def char2pinyin(sent):
            """
            Transform Chars in Sent to Pinyin
            1. Multiple Pinyin for a Char
            2. Unk Char remain the same
            :return [ [pinyin1, pinyin2, ...], [unk_char] ]
            """
            res = []
            for char in sent:
                if char in self.pinyin_dict:
                    res.append(self.pinyin_dict[char])
                elif len(char) > 0 and char[0].islower():
                    char = char[0].upper() + char[1:]
                    #    yun OS
                    # => Yun OS
                    res.append(self.pinyin_dict[char])
                else:
                    res.append([char])
            return res

        pinyin_sa = char2pinyin(char_sa)
        pinyin_sb = char2pinyin(char_sb)

        def similarity(pinyin_sa, pinyin_sb):
            """
            sa: [ [pinyin1, pinyin2, ...], [unk_char] ]
            :return overlap(sa, sb) / |sa|
            """
            pinyin_sb = sum(pinyin_sb, [])
            overlap = 0
            for pinyin_list in pinyin_sa:
                for pinyin in pinyin_list:
                    if pinyin in pinyin_sb:
                        overlap += 1
                        break
            overlap_rate = 1. * overlap / len(pinyin_sa) if len(pinyin_sa) > 0 else 0.0
            return overlap_rate

        features = [ similarity(pinyin_sa, pinyin_sb) ]
        infos = ['pinyin', pinyin_sa, pinyin_sb]
        return features, infos
