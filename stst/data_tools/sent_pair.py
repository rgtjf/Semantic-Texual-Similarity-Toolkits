# coding: utf8
from __future__ import print_function
from __future__ import unicode_literals

from stst import utils


class SentPair(object):

    def __init__(self, json_pair):
        """
        :param sa: list of words
        :param sb: list of words
        :param score: float
        """
        self.sa = json_pair['sa']['parsed']
        self.sb = json_pair['sb']['parsed']
        self.score = json_pair['label']

    def get_word(self):
        sa = [item['word'] for item in self.sa]
        sb = [item['word'] for item in self.sb]
        return sa, sb

    def get_pos(self):
        sa = [item['pos'] for item in self.sa]
        sb = [item['pos'] for item in self.sb]
        return sa, sb

    def get_expand_word(self):
        # pos_sa, pos_sb = self.get_pos()
        word_sa, word_sb = self.get_word()

        if all(item['ner'] != '' or item['word'].strip() == '' for item in self.sa):
            word_sa += ['是', '什么']
            print(' '.join(word_sa))

        if all(item['ner'] != '' or item['word'].strip() == '' for item in self.sb):
            word_sb += ['是', '什么']
            print(' '.join(word_sb))
        return word_sa, word_sb

    def get_char(self):
        word_sa, word_sb = self.get_word()
        char_sa = utils.word2char(word_sa)
        char_sb = utils.word2char(word_sb)
        return char_sa, char_sb

    def get_ner(self):
        sa = []
        for item in self.sa:
            if item['ner'] != '':
                sa.append(item['ner'])
            else:
                pass
                # sa.append(item['word'])

        sb = []
        for item in self.sb:
            if item['ner'] != '':
                sb.append(item['ner'])
            else:
                pass
                # sb.append(item['word'])
        return sa, sb

    def get_pair(self, type):
        if type == 'word':
            sa, sb = self.get_word()
        elif type == 'char':
            sa, sb =self.get_char()
        elif type == 'ner':
            sa, sb = self.get_ner()
        elif type == 'word+':
            sa, sb = self.get_expand_word()
        else:
            raise NotImplementedError
        return sa, sb

    def get_score(self):
        return self.score

    def get_instance_string(self):
        sa, sb = self.get_word()
        sa, sb = ' '.join(sa), ' '.join(sb)
        instance_string = '{:.3f}\t{}\t{}'.format(self.score, sa, sb)
        return instance_string

    def __call__(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    pass