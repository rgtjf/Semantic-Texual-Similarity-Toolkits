from stst.data import dict_utils
from stst.modules.features import Feature


class NegativeFeature(Feature):
    def extract(self, train_instance):
        negation = dict_utils.DictLoader().load_dict('negation_terms')
        lemma_sa, lemma_sb = train_instance.get_word(type='lemma', lower=True)
        na = sum([1 if w in negation else 0 for w in lemma_sa])
        nb = sum([1 if w in negation else 0 for w in lemma_sb])

        features = [(na - nb) % 2]
        infos = [na, nb]
        return features, infos


class EnNegativeFeature(Feature):
    def __init__(self, penalty, **kwargs):
        super(EnNegativeFeature, self).__init__(**kwargs)
        self.load = False
        self.penalty = penalty

    def extract(self, train_instance):
        negation = dict_utils.DictLoader().load_dict('negation_terms')
        lemma_sa, lemma_sb = train_instance.get_word(type='lemma', lower=True)
        na = sum([1 if w in negation else 0 for w in lemma_sa])
        nb = sum([1 if w in negation else 0 for w in lemma_sb])

        if (na - nb) % 2 != 0:
            score = self.penalty
        else:
            score = 0.0
        features = [score]
        infos = [na, nb]
        return features, infos
