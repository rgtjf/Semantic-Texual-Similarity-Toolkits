from stst.lib.word_aligner.aligner import *
from stst.lib.pycorenlp.corenlp_utils import *

sa = "Two green and white trains sitting on the tracks."
sb = "Two green and white trains on tracks."

sa = "Four men died in an accident."
sb = "4 people are dead from a collision."
nlp = StanfordNLP()
parse_sa, parse_sb = nlp.parse(sa), nlp.parse(sb)

# print(parse_sa)

features, infos = align_feats(parse_sa, parse_sb)

print(features, infos)
