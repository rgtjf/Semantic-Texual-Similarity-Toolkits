# coding: utf8
from __future__ import print_function

from nltk.corpus import wordnet as wn

from stst import utils
from stst.data.dict_utils import DictLoader


class SentPair(object):
    def __init__(self, parse_json):
        """
        :param  sa, sb
        :param  score(0,5) -> score / 5.0
        :return SentPair: tuple, (sa, sb)
        """
        parse_sa, parse_sb, score = parse_json
        self.parse_sa = parse_sa  # str
        self.parse_sb = parse_sb  # str
        self.score = float(score) / 5.0  # float

    def get_preprocess(self, **kwargs):
        word_sa, word_sb = self.get_word(type='lemma', stopwords=True, lower=True)
        pos_sa, pos_sb = self.get_word(type='pos', stopwords=True)

        if len(word_sa) == 0 or len(word_sb) == 0:
            word_sa, word_sb = self.get_word(type='lemma', stopwords=False, lower=True)
            pos_sa, pos_sb = self.get_word(type='pos', stopwords=False)

        pos_sa = map(utils.pos2tag, pos_sa)
        pos_sb = map(utils.pos2tag, pos_sb)
        # pos_sa, pos_sb = self.get_pos_tag(stopwords=False, lower=True)
        pos_sa = zip(word_sa, pos_sa)
        pos_sb = zip(word_sb, pos_sb)

        def transform(items):
            sent = []
            for item in items:
                fields = item
                lemma = fields[0]
                pos = fields[1]
                sent.append([lemma, pos, False])
            return sent

        def algin(pos_sa, pos_sb):

            sent1, sent2 = transform(pos_sa), transform(pos_sb)
            for idx1 in range(len(sent1)):
                if sent1[idx1][2]:
                    continue
                for idx2 in range(len(sent2)):
                    if sent1[idx1][0] == sent2[idx2][0]:
                        sent1[idx1][2] = True
                        sent2[idx2][2] = True
                        break
                    else:
                        synonyms = get_synonyms((sent2[idx2][0], sent2[idx2][1]))
                        if sent1[idx1][0] in synonyms:
                            print('synonym : %s %s' % (sent1[idx1][0], sent2[idx2][0]))
                            sent1[idx1][2] = True
                            sent2[idx2][2] = True
                            sent1[idx1][0] = sent2[idx2][0]
            sent1 = [item[0] for item in sent1]
            sent2 = [item[0] for item in sent2]
            return sent1, sent2

        sa, sb = algin(pos_sa, pos_sb)
        return sa, sb

    def get_word(self, **kwargs):
        """
        :param kwargs: type=word/lemma/pos/ner, stopwords=True/False
        :return:
        """
        sa, sb = [], []

        if kwargs.get('stopwords', False) is True:

            tokens = self.parse_sa["sentences"][0]["tokens"]
            sa = [ token[kwargs["type"]] for token in tokens if token['word'].lower() not in DictLoader().load_dict('stopwords') ]

            tokens = self.parse_sb["sentences"][0]["tokens"]
            sb = [ token[kwargs["type"]] for token in tokens if token['word'].lower() not in DictLoader().load_dict('stopwords') ]

            if len(sa) == 0 or len(sb) == 0:
                tokens = self.parse_sa["sentences"][0]["tokens"]
                sa = [token[kwargs["type"]] for token in tokens]

                tokens = self.parse_sb["sentences"][0]["tokens"]
                sb = [token[kwargs["type"]] for token in tokens]

        else:
            tokens = self.parse_sa["sentences"][0]["tokens"]
            sa = [ token[kwargs["type"]] for token in tokens ]

            tokens = self.parse_sb["sentences"][0]["tokens"]
            sb = [ token[kwargs["type"]] for token in tokens ]

        if 'lower' in kwargs and kwargs['lower'] is True:
            sa = [ w.lower() for w in sa ]
            sb = [ w.lower() for w in sb ]

        return sa, sb

    def get_parse(self):
        return self.parse_sa, self.parse_sb

    def get_char(self, **kwargs):
        lemma_stopwords_sa, lemma_stopwords_sb = self.get_word(type='lemma', **kwargs)
        sa = '^' + '#'.join(lemma_stopwords_sa) + '$'
        sb = '^' + '#'.join(lemma_stopwords_sb) + '$'

        return sa, sb

    def get_ner(self):
        tokens = self.parse_sa["sentences"][0]["tokens"]
        ner_sa = [ [token['lemma'], token['ner']] for token in tokens]

        tokens = self.parse_sb["sentences"][0]["tokens"]
        ner_sb = [ [token['lemma'], token['ner']] for token in tokens]

        return ner_sa, ner_sb

    def get_pos(self):
        tokens = self.parse_sa["sentences"][0]["tokens"]
        pos_sa = [[token['lemma'], token['pos']] for token in tokens if
                  token['word'].lower() not in DictLoader().load_dict('stopwords')]

        tokens = self.parse_sb["sentences"][0]["tokens"]
        pos_sb = [[token['lemma'], token['pos']] for token in tokens if
                  token['word'].lower() not in DictLoader().load_dict('stopwords')]

        return pos_sa, pos_sb

    def get_pos_tag(self, stopwords=True):
        if stopwords:
            tokens = self.parse_sa["sentences"][0]["tokens"]
            pos_sa = [[token['lemma'], utils.pos2tag(token['pos'])] for token in tokens if
                      token['word'].lower() not in DictLoader().load_dict('stopwords')]

            tokens = self.parse_sb["sentences"][0]["tokens"]
            pos_sb = [[token['lemma'], utils.pos2tag(token['pos'])] for token in tokens if
                      token['word'].lower() not in DictLoader().load_dict('stopwords')]
        else:
            tokens = self.parse_sa["sentences"][0]["tokens"]
            pos_sa = [[token['lemma'], utils.pos2tag(token['pos'])] for token in tokens]

            tokens = self.parse_sb["sentences"][0]["tokens"]
            pos_sb = [[token['lemma'], utils.pos2tag(token['pos'])] for token in tokens]

        return pos_sa, pos_sb

    def get_score(self):
        return self.score

    def get_instance_string(self):
        sa, sb = self.get_word(type='word')
        sa, sb = ' '.join(sa), ' '.join(sb)
        instance_string = str(self.score) + '\t' + sa  + '\t' + sb
        return instance_string

    def get_dependency(self):
        dep_sa = self.parse_sa['sentences'][0]['basic-dependencies']
        dep_sb = self.parse_sb['sentences'][0]['basic-dependencies']
        lemma_sa, lemma_sb = self.get_word(type='lemma', lower=True)
        deps = []
        for dep in dep_sa:
            rel = dep['dep']
            dependent = 'ROOT' if dep['dependent'] == 0 else lemma_sa[dep['dependent']-1]
            governor = 'ROOT' if dep['governor'] == 0 else lemma_sa[dep['governor']-1]
            deps.append((rel, governor, dependent))
        dep_sa = deps

        deps = []
        for dep in dep_sb:
            rel = dep['dep']
            dependent = 'ROOT' if dep['dependent'] == 0 else lemma_sb[dep['dependent'] - 1]
            governor = 'ROOT' if dep['governor'] == 0 else lemma_sb[dep['governor'] - 1]
            deps.append((rel, governor, dependent))
        dep_sb = deps

        return dep_sa, dep_sb


    def __call__(self, *args, **kwargs):
        # [lemma, stem, stopwords]
        pass



# input format:(word, pos)
# output format: set of word
def get_synonyms(word_with_pos):
    if word_with_pos[1] == '#':
        return []
    synsets = wn.synsets(word_with_pos[0], word_with_pos[1])
    synonyms = []
    for synset in synsets:
        name = synset.name().split('.')
        synonym = (name[0], name[1])
        if not synonym == word_with_pos:
            synonyms.append(synonym)
    return set([word for word, pos in synonyms])

if __name__ == '__main__':
    ss = [{"sentences": [{"tokens": [{"index": 1, "word": "In", "lemma": "in", "after": " ", "pos": "IN", "characterOffsetEnd": 2, "characterOffsetBegin": 0, "originalText": "In", "ner": "O", "before": ""}, {"index": 2, "word": "the", "lemma": "the", "after": " ", "pos": "DT", "characterOffsetEnd": 6, "characterOffsetBegin": 3, "originalText": "the", "ner": "O", "before": " "}, {"index": 3, "word": "US", "lemma": "US", "after": "", "pos": "NNP", "characterOffsetEnd": 9, "characterOffsetBegin": 7, "originalText": "US", "ner": "LOCATION", "before": " "}, {"index": 4, "word": ",", "lemma": ",", "after": " ", "pos": ",", "characterOffsetEnd": 10, "characterOffsetBegin": 9, "originalText": ",", "ner": "O", "before": ""}, {"index": 5, "word": "it", "lemma": "it", "after": " ", "pos": "PRP", "characterOffsetEnd": 13, "characterOffsetBegin": 11, "originalText": "it", "ner": "O", "before": " "}, {"index": 6, "word": "will", "lemma": "will", "after": " ", "pos": "MD", "characterOffsetEnd": 18, "characterOffsetBegin": 14, "originalText": "will", "ner": "O", "before": " "}, {"index": 7, "word": "depend", "lemma": "depend", "after": " ", "pos": "VB", "characterOffsetEnd": 25, "characterOffsetBegin": 19, "originalText": "depend", "ner": "O", "before": " "}, {"index": 8, "word": "on", "lemma": "on", "after": " ", "pos": "IN", "characterOffsetEnd": 28, "characterOffsetBegin": 26, "originalText": "on", "ner": "O", "before": " "}, {"index": 9, "word": "the", "lemma": "the", "after": " ", "pos": "DT", "characterOffsetEnd": 32, "characterOffsetBegin": 29, "originalText": "the", "ner": "O", "before": " "}, {"index": 10, "word": "school", "lemma": "school", "after": "", "pos": "NN", "characterOffsetEnd": 39, "characterOffsetBegin": 33, "originalText": "school", "ner": "O", "before": " "}, {"index": 11, "word": ".", "lemma": ".", "after": "", "pos": ".", "characterOffsetEnd": 40, "characterOffsetBegin": 39, "originalText": ".", "ner": "O", "before": ""}], "index": 0, "basic-dependencies": [{"dep": "ROOT", "dependent": 7, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "depend"}, {"dep": "case", "dependent": 1, "governorGloss": "US", "governor": 3, "dependentGloss": "In"}, {"dep": "det", "dependent": 2, "governorGloss": "US", "governor": 3, "dependentGloss": "the"}, {"dep": "nmod", "dependent": 3, "governorGloss": "depend", "governor": 7, "dependentGloss": "US"}, {"dep": "punct", "dependent": 4, "governorGloss": "depend", "governor": 7, "dependentGloss": ","}, {"dep": "nsubj", "dependent": 5, "governorGloss": "depend", "governor": 7, "dependentGloss": "it"}, {"dep": "aux", "dependent": 6, "governorGloss": "depend", "governor": 7, "dependentGloss": "will"}, {"dep": "case", "dependent": 8, "governorGloss": "school", "governor": 10, "dependentGloss": "on"}, {"dep": "det", "dependent": 9, "governorGloss": "school", "governor": 10, "dependentGloss": "the"}, {"dep": "nmod", "dependent": 10, "governorGloss": "depend", "governor": 7, "dependentGloss": "school"}, {"dep": "punct", "dependent": 11, "governorGloss": "depend", "governor": 7, "dependentGloss": "."}], "parse": "(ROOT\n  (S\n    (PP (IN In)\n      (NP (DT the) (NNP US)))\n    (, ,)\n    (NP (PRP it))\n    (VP (MD will)\n      (VP (VB depend)\n        (PP (IN on)\n          (NP (DT the) (NN school)))))\n    (. .)))", "collapsed-dependencies": [{"dep": "ROOT", "dependent": 7, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "depend"}, {"dep": "case", "dependent": 1, "governorGloss": "US", "governor": 3, "dependentGloss": "In"}, {"dep": "det", "dependent": 2, "governorGloss": "US", "governor": 3, "dependentGloss": "the"}, {"dep": "nmod:in", "dependent": 3, "governorGloss": "depend", "governor": 7, "dependentGloss": "US"}, {"dep": "punct", "dependent": 4, "governorGloss": "depend", "governor": 7, "dependentGloss": ","}, {"dep": "nsubj", "dependent": 5, "governorGloss": "depend", "governor": 7, "dependentGloss": "it"}, {"dep": "aux", "dependent": 6, "governorGloss": "depend", "governor": 7, "dependentGloss": "will"}, {"dep": "case", "dependent": 8, "governorGloss": "school", "governor": 10, "dependentGloss": "on"}, {"dep": "det", "dependent": 9, "governorGloss": "school", "governor": 10, "dependentGloss": "the"}, {"dep": "nmod:on", "dependent": 10, "governorGloss": "depend", "governor": 7, "dependentGloss": "school"}, {"dep": "punct", "dependent": 11, "governorGloss": "depend", "governor": 7, "dependentGloss": "."}], "collapsed-ccprocessed-dependencies": [{"dep": "ROOT", "dependent": 7, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "depend"}, {"dep": "case", "dependent": 1, "governorGloss": "US", "governor": 3, "dependentGloss": "In"}, {"dep": "det", "dependent": 2, "governorGloss": "US", "governor": 3, "dependentGloss": "the"}, {"dep": "nmod:in", "dependent": 3, "governorGloss": "depend", "governor": 7, "dependentGloss": "US"}, {"dep": "punct", "dependent": 4, "governorGloss": "depend", "governor": 7, "dependentGloss": ","}, {"dep": "nsubj", "dependent": 5, "governorGloss": "depend", "governor": 7, "dependentGloss": "it"}, {"dep": "aux", "dependent": 6, "governorGloss": "depend", "governor": 7, "dependentGloss": "will"}, {"dep": "case", "dependent": 8, "governorGloss": "school", "governor": 10, "dependentGloss": "on"}, {"dep": "det", "dependent": 9, "governorGloss": "school", "governor": 10, "dependentGloss": "the"}, {"dep": "nmod:on", "dependent": 10, "governorGloss": "depend", "governor": 7, "dependentGloss": "school"}, {"dep": "punct", "dependent": 11, "governorGloss": "depend", "governor": 7, "dependentGloss": "."}]}]},
         {"sentences": [{"tokens": [{"index": 1, "word": "It", "lemma": "it", "after": " ", "pos": "PRP", "characterOffsetEnd": 2, "characterOffsetBegin": 0, "originalText": "It", "ner": "O", "before": ""}, {"index": 2, "word": "really", "lemma": "really", "after": " ", "pos": "RB", "characterOffsetEnd": 9, "characterOffsetBegin": 3, "originalText": "really", "ner": "O", "before": " "}, {"index": 3, "word": "depends", "lemma": "depend", "after": " ", "pos": "VBZ", "characterOffsetEnd": 17, "characterOffsetBegin": 10, "originalText": "depends", "ner": "O", "before": " "}, {"index": 4, "word": "on", "lemma": "on", "after": " ", "pos": "IN", "characterOffsetEnd": 20, "characterOffsetBegin": 18, "originalText": "on", "ner": "O", "before": " "}, {"index": 5, "word": "the", "lemma": "the", "after": " ", "pos": "DT", "characterOffsetEnd": 24, "characterOffsetBegin": 21, "originalText": "the", "ner": "O", "before": " "}, {"index": 6, "word": "school", "lemma": "school", "after": " ", "pos": "NN", "characterOffsetEnd": 31, "characterOffsetBegin": 25, "originalText": "school", "ner": "O", "before": " "}, {"index": 7, "word": "and", "lemma": "and", "after": " ", "pos": "CC", "characterOffsetEnd": 35, "characterOffsetBegin": 32, "originalText": "and", "ner": "O", "before": " "}, {"index": 8, "word": "the", "lemma": "the", "after": " ", "pos": "DT", "characterOffsetEnd": 39, "characterOffsetBegin": 36, "originalText": "the", "ner": "O", "before": " "}, {"index": 9, "word": "program", "lemma": "program", "after": "", "pos": "NN", "characterOffsetEnd": 47, "characterOffsetBegin": 40, "originalText": "program", "ner": "O", "before": " "}, {"index": 10, "word": ".", "lemma": ".", "after": "", "pos": ".", "characterOffsetEnd": 48, "characterOffsetBegin": 47, "originalText": ".", "ner": "O", "before": ""}], "index": 0, "basic-dependencies": [{"dep": "ROOT", "dependent": 3, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "depends"}, {"dep": "nsubj", "dependent": 1, "governorGloss": "depends", "governor": 3, "dependentGloss": "It"}, {"dep": "advmod", "dependent": 2, "governorGloss": "depends", "governor": 3, "dependentGloss": "really"}, {"dep": "case", "dependent": 4, "governorGloss": "school", "governor": 6, "dependentGloss": "on"}, {"dep": "det", "dependent": 5, "governorGloss": "school", "governor": 6, "dependentGloss": "the"}, {"dep": "nmod", "dependent": 6, "governorGloss": "depends", "governor": 3, "dependentGloss": "school"}, {"dep": "cc", "dependent": 7, "governorGloss": "school", "governor": 6, "dependentGloss": "and"}, {"dep": "det", "dependent": 8, "governorGloss": "program", "governor": 9, "dependentGloss": "the"}, {"dep": "conj", "dependent": 9, "governorGloss": "school", "governor": 6, "dependentGloss": "program"}, {"dep": "punct", "dependent": 10, "governorGloss": "depends", "governor": 3, "dependentGloss": "."}], "parse": "(ROOT\n  (S\n    (NP (PRP It))\n    (ADVP (RB really))\n    (VP (VBZ depends)\n      (PP (IN on)\n        (NP\n          (NP (DT the) (NN school))\n          (CC and)\n          (NP (DT the) (NN program)))))\n    (. .)))", "collapsed-dependencies": [{"dep": "ROOT", "dependent": 3, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "depends"}, {"dep": "nsubj", "dependent": 1, "governorGloss": "depends", "governor": 3, "dependentGloss": "It"}, {"dep": "advmod", "dependent": 2, "governorGloss": "depends", "governor": 3, "dependentGloss": "really"}, {"dep": "case", "dependent": 4, "governorGloss": "school", "governor": 6, "dependentGloss": "on"}, {"dep": "det", "dependent": 5, "governorGloss": "school", "governor": 6, "dependentGloss": "the"}, {"dep": "nmod:on", "dependent": 6, "governorGloss": "depends", "governor": 3, "dependentGloss": "school"}, {"dep": "cc", "dependent": 7, "governorGloss": "school", "governor": 6, "dependentGloss": "and"}, {"dep": "det", "dependent": 8, "governorGloss": "program", "governor": 9, "dependentGloss": "the"}, {"dep": "conj:and", "dependent": 9, "governorGloss": "school", "governor": 6, "dependentGloss": "program"}, {"dep": "punct", "dependent": 10, "governorGloss": "depends", "governor": 3, "dependentGloss": "."}], "collapsed-ccprocessed-dependencies": [{"dep": "ROOT", "dependent": 3, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "depends"}, {"dep": "nsubj", "dependent": 1, "governorGloss": "depends", "governor": 3, "dependentGloss": "It"}, {"dep": "advmod", "dependent": 2, "governorGloss": "depends", "governor": 3, "dependentGloss": "really"}, {"dep": "case", "dependent": 4, "governorGloss": "school", "governor": 6, "dependentGloss": "on"}, {"dep": "det", "dependent": 5, "governorGloss": "school", "governor": 6, "dependentGloss": "the"}, {"dep": "nmod:on", "dependent": 6, "governorGloss": "depends", "governor": 3, "dependentGloss": "school"}, {"dep": "cc", "dependent": 7, "governorGloss": "school", "governor": 6, "dependentGloss": "and"}, {"dep": "det", "dependent": 8, "governorGloss": "program", "governor": 9, "dependentGloss": "the"}, {"dep": "nmod:on", "dependent": 9, "governorGloss": "depends", "governor": 3, "dependentGloss": "program"}, {"dep": "conj:and", "dependent": 9, "governorGloss": "school", "governor": 6, "dependentGloss": "program"}, {"dep": "punct", "dependent": 10, "governorGloss": "depends", "governor": 3, "dependentGloss": "."}]}]}, 3.0]
    sa = ss[0]
    sb = ss[1]
    ss = SentPair(ss)
    print(ss.get_preprocess())

"""
print(json.dumps(parse_sa, indent=2))
{
   "sentences": [
       {
           "tokens": [
               {
                   "index": 1,
                   "word": "I",
                   "lemma": "I",
                   "after": " ",
                   "pos": "PRP",
                   "characterOffsetEnd": 1,
                   "characterOffsetBegin": 0,
                   "originalText": "I",
                   "ner": "O",
                   "before": ""
               },
               .....
               {
                   "index": 8,
                   "word": ".",
                   "lemma": ".",
                   "after": "",
                   "pos": ".",
                   "characterOffsetEnd": 30,
                   "characterOffsetBegin": 29,
                   "originalText": ".",
                   "ner": "O",
                   "before": ""
               }
           ],
           "basic-dependencies": [
               {
                   "dep": "ROOT",
                   "dependent": 2,
                   "governorGloss": "ROOT",
                   "governor": 0,
                   "dependentGloss": "love"
               },
               ...
               {
                   "dep": "punct",
                   "dependent": 8,
                   "governorGloss": "love",
                   "governor": 2,
                   "dependentGloss": "."
               }
           ],
           "parse": "(ROOT\n  (S\n    (NP (PRP I))\n    (VP (VBP love)\n      (NP\n        (NP (NNP China) (. .))\n        (SBAR\n          (S\n            (NP (PRP I))\n            (VP (VBP love)\n              (NP (NNP Shanghai)))))))\n    (. .)))",
       }
   ]
}
"""
