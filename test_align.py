# coding: utf8
from .corenlp_utils import StanfordNLP

from aligner import align_feats

sa = "Two green and white trains sitting on the tracks."
sb = "Two green and white trains on tracks."

sa = "Four men died in an accident."
sb = "4 people are dead from a collision."
nlp = StanfordNLP(server_url='http://localhost:9000')
parse_sa, parse_sb = nlp.parse(sa), nlp.parse(sb)

features, infos = align_feats(parse_sa, parse_sb)

print(features)
print(infos[0])
print(infos[1])
