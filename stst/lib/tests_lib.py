# coding: utf8
from __future__ import print_function


def test_kernel():
    from stst.lib.kernel.vector_kernel import euclidean_distance, chebyshev_distance, cosine_distance
    import numpy as np
    v1 = np.array([0, 3], dtype=np.float32)
    v2 = np.array([4, 0], dtype=np.float32)
    print(euclidean_distance(v1, v2))
    print(chebyshev_distance(v1, v2))
    print(cosine_distance(v1, v2))

def test_pycorenlp():
    from stst.lib.pycorenlp.corenlp_utils import StanfordNLP
    import json
    nlp = StanfordNLP()
    parsetext = nlp.parse('I love China.')
    print(json.dumps(parsetext, indent=2))

def test_sentence_similarity():
    from stst.lib.sentence_similarity.short_sentence_similarity import similarity
    sentence_pairs = [
        ["I like that bachelor.", "I like that unmarried man.", 0.561]
    ]
    for sent_pair in sentence_pairs:
        print("%s\t%s\t%.3f\t%.3f\t%.3f" % (sent_pair[0], sent_pair[1], sent_pair[2],
                                            similarity(sent_pair[0], sent_pair[1], False),
                                            similarity(sent_pair[0], sent_pair[1], True)))

test_sentence_similarity()
