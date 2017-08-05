# coding: utf8
from __future__ import print_function

from stst import utils


class SentPair(object):

    def __init__(self, json_pair):
        """
        :param sa: list of words
        :param sb: list of words
        :param score: float
        """
        self.sa = json_pair['sa']
        self.sb = json_pair['sb']
        self.score = json_pair['score']

    def get_word(self):
        return self.sa, self.sb

    def get_char(self):
        char_sa = utils.word2char(self.sa)
        char_sb = utils.word2char(self.sb)
        return char_sa, char_sb

    def get_pair(self, type):
        if type == 'word':
            sa, sb = self.get_word()
        elif type == 'char':
            sa, sb =self.get_char()
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