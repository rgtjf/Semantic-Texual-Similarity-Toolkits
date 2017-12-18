# coding: utf8
from __future__ import print_function
import unittest


class TestLibs(unittest.TestCase):

    def test_kernel(self):
        from kernel.vector_kernel import euclidean_distance, chebyshev_distance, cosine_distance
        import numpy as np
        v1 = np.array([0, 3], dtype=np.float32)
        v2 = np.array([4, 0], dtype=np.float32)
        print(euclidean_distance(v1, v2))
        print(chebyshev_distance(v1, v2))
        print(cosine_distance(v1, v2))

    def test_sentence_similarity(self):
        from sentence_similarity.short_sentence_similarity import similarity
        sentence_pairs = [
            ["I like that bachelor.", "I like that unmarried man.", 0.561]
        ]
        for sent_pair in sentence_pairs:
            print("%s\t%s\t%.3f\t%.3f\t%.3f" % (sent_pair[0], sent_pair[1], sent_pair[2],
                                                similarity(sent_pair[0], sent_pair[1], False),
                                                similarity(sent_pair[0], sent_pair[1], True)))

    def test_aligner(self):
        # coding: utf8
        from word_aligner.corenlp_utils import StanfordNLP
        from word_aligner.aligner import align_feats

        sa = "Two green and white trains sitting on the tracks."
        sb = "Two green and white trains on tracks."

        sa = "Four men died in an accident."
        sb = "4 people are dead from a collision."
        nlp = StanfordNLP(server_url='http://precision:9000')
        parse_sa, parse_sb = nlp.parse(sa), nlp.parse(sb)

        features, infos = align_feats(parse_sa, parse_sb)

        print(features)
        print(infos[0])
        print(infos[1])


if __name__ == '__main__':
    unittest.main()
