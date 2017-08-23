# coding: utf8
from __future__ import print_function
from __future__ import unicode_literals

from stst import utils


class SentPair(object):

    def __init__(self, json_pair):
        """
         {"text": "你可以做哪些事啊", "id": "245", "parsed": [{"index": 1, "head": 2, "word": "你", "pos": "PN", "label": "SUB", "ner": ""}, {"index": 2, "head": 0, "word": "可以", "pos": "VV", "label": "ROOT", "ner": ""}, {"index": 3, "head": 2, "word": "做", "pos": "VV", "label": "VC", "ner": ""}, {"index": 4, "head": 5, "word": "哪些", "pos": "DT", "label": "NMOD", "ner": ""}, {"index": 5, "head": 3, "word": "事", "pos": "NN", "label": "OBJ", "ner": ""}, {"index": 6, "head": 2, "word": "啊", "pos": "SP", "label": "DEP", "ner": ""}]}
        :param sa: list of words
        :param sb: list of words
        :param score: float
        """
        self.sa = json_pair['sa']['parsed']
        self.sb = json_pair['sb']['parsed']
        self.score = json_pair['label']
        self.sa_question_type = json_pair['sa']['question_type']
        self.sb_question_type = json_pair['sb']['question_type']

    def get_word(self):
        sa = [item['word'] for item in self.sa]
        sb = [item['word'] for item in self.sb]
        return sa, sb

    def get_pos(self):
        sa = [item['pos'] for item in self.sa]
        sb = [item['pos'] for item in self.sb]
        return sa, sb

    def get_question_type(self):
        return self.sa_question_type, self.sb_question_type

    def get_expand_word(self):
        # pos_sa, pos_sb = self.get_pos()
        word_sa, word_sb = self.get_word()

        if all(item['ner'] != '' for item in self.sa):
            word_sa += ['是', '什么']
            # print(' '.join(word_sa))

        if all(item['ner'] != '' for item in self.sb):
            word_sb += ['是', '什么']
            # print(' '.join(word_sb))
        return word_sa, word_sb

    def get_char(self):
        word_sa, word_sb = self.get_word()
        char_sa = utils.word2char(word_sa)
        char_sb = utils.word2char(word_sb)
        return char_sa, char_sb

    def get_list(self, type):
        """
        Args:
            type: word pos ner
        """
        sa = [item[type] for item in self.sa]
        sb = [item[type] for item in self.sb]
        return sa, sb


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

    def get_pos_dep(self):
        """
        <rel, pos, head_pos>
        """
        def get_triple(parse_sa, pos_sa):
            sa = []
            for word in parse_sa:
                rel = word['label']
                if word['head'] == 0:
                    head_word = 'ROOT'
                else:
                    head_word = pos_sa[word['head'] - 1]
                sa.append((rel, word['pos'], head_word))
            return sa

        pos_sa, pos_sb = self.get_pos()

        pos_dep_sa = get_triple(self.sa, pos_sa)
        pos_dep_sb = get_triple(self.sb, pos_sb)

        return pos_dep_sa, pos_dep_sb

    def get_word_dep(self):
        """
        <rel, word, head_word>
        """

        def get_triple(parse_sa, word_sa):
            sa = []
            for word in parse_sa:
                rel = word['label']
                if word['head'] == 0:
                    head_word = 'ROOT'
                else:
                    head_word = word_sa[word['head']-1]
                sa.append((rel, word['word'], head_word))
            return sa

        word_sa, word_sb = self.get_word()
        word_dep_sa = get_triple(self.sa, word_sa)
        word_dep_sb = get_triple(self.sb, word_sb)
        return word_dep_sa, word_dep_sb

    def get_pair(self, type):
        if type == 'word':
            sa, sb = self.get_word()
        elif type == 'char':
            sa, sb =self.get_char()
        elif type == 'ner':
            sa, sb = self.get_ner()
        elif type == 'word+':
            sa, sb = self.get_expand_word()
        elif type == 'word_dep':
            sa, sb = self.get_word_dep()
        elif type == 'pos_dep':
            sa, sb = self.get_pos_dep()
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