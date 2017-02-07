from features import Feature
import utils


class NERMatchFeature(Feature):
    def extract(self, train_instance):
        ner_sa, ner_sb = train_instance.get_ner()
        features, infos = self.core_extract(ner_sa, ner_sb)
        return features, infos

    def core_extract(self, sa, sb):

        ner_sa = [ner for lemma, ner in sa]
        ner_sb = [ner for lemma, ner in sb]

        features = []
        infos = []

        """ ner_type position """
        # for ner_type in ['LOCATION', 'Person', 'Organization', 'Money', 'Percent', 'Date', 'Time']:
        #     position_sa = 0
        #     position_sb = 0
        #     for idx, ner in enumerate(ner_sa):
        #         if ner == ner_type:
        #             position_sa = idx + 1
        #             break
        #     for idx, ner in enumerate(ner_sb):
        #         if ner == ner_type:
        #             position_sb = idx + 1
        #             break
        #     features += [position_sa, position_sb, (abs(position_sa - position_sb))]
        # features += []
        """ ner_type appear """
        for ner_type in ['LOCATION', 'Person', 'Organization', 'Money', 'Percent', 'Date', 'Time']:
            score = 0.2
            if ner_type in ner_sa and ner_type in ner_sb:
                score = 1.0
            elif ner_type in ner_sa or ner_type in ner_sb:
                score = 0.2
            else:
                score = 0.5
            features.append(score)
        return features, infos


class NERWordFeature(Feature):
    # def extract(self, train_instance):
    #     """ ner_type word match """
    #     word_sa, word_sb = train_instance.get_word(type='lemma', lower=True)
    #     ner_sa, ner_sb = train_instance.get_word(type='ner')
    #
    #     word_sa = [word for word, ner in zip(word_sa, ner_sa) if ner != 'O']
    #     word_sb = [word for word, ner in zip(word_sa, ner_sa) if ner != 'O']
    #     A_inter_B = len([word for word in word_sa if word in word_sb])
    #     # features, infos = utils.sequence_match_features(word_sa, word_sb)
    #     features = [A_inter_B]
    #     infos = [word_sa, word_sb]
    #     return features, infos

    def extract(self, train_instance):
        """ ner_type word match """
        word_sa, word_sb = train_instance.get_word(type='lemma', lower=True)
        ner_sa, ner_sb = train_instance.get_word(type='ner')

        word_sa = [word for word, ner in zip(word_sa, ner_sa) if ner != 'O']
        word_sb = [word for word, ner in zip(word_sa, ner_sa) if ner != 'O']
        A_inter_B = len([word for word in word_sa if word in word_sb])
        # features, infos = utils.sequence_match_features(word_sa, word_sb)
        features = [A_inter_B]
        infos = [word_sa, word_sb]
        return features, infos

class NERVectorFeature(Feature):
    def extract_information(self, train_instances):
        self.ners = ['LOCATION', 'Person', 'Organization', 'Money', 'Percent', 'Date', 'Time']

    def extract(self, train_instance):
        """Important question
        if ner not show at the same time or show at the same time,
        the distance may be same.
        """
        pass


class NEROverlapFeature(Feature):
    def extract(self, train_instance):
        word_sa, word_sb = train_instance.get_word(type='lemma', lower=True)
        ner_sa, ner_sb = train_instance.get_word(type='ner')


        """ ner_type appear """
        for ner_type in ['LOCATION', 'Person', 'Organization', 'Money', 'Percent', 'Date', 'Time']:
            ner_type_sa = []
            for word, ner in zip(word_sa, ner_sa):
                if ner == ner_type:
                    ner_type_sa.append(word)

            ner_type_sb = []
            for word, ner in zip(word_sb, ner_sb):
                if ner == ner_type:
                    ner_type_sb.append(word)

        #     features +=
        #
        # return features, infos
