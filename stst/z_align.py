from lib.word_aligner.aligner import *
from lib.pycorenlp.coreNlpUtil import *

sa = "Two green and white trains sitting on the tracks."
sb = "Two green and white trains on tracks."

parse_sa, parse_sb = nlp.parse(sa), nlp.parse(sb)

print(parse_sa)

features, infos = align_feats(parse_sa, parse_sb)

print(features, infos)