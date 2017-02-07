import utils
from features import Feature
import nltk, math


class DependencyPositionFeature(Feature):
    def make_triple_set(self, train_instances, min_cnt=1, stopwords=False):
        doc_num = 0
        word_list = []
        for train_instance in train_instances:
            lemma_sa, lemma_sb = train_instance.get_word(type='lemma', stopwords=stopwords, lower=True)
            word_list += lemma_sa
            word_list += lemma_sb
            doc_num += 2

        vdist = nltk.FreqDist(word_list)
        idf_dict = {}
        good_keys = [v for v in vdist.keys() if vdist[v] >= min_cnt]

        for key in good_keys:
            idf_dict[key] = vdist[key]

        for key in idf_dict.keys():
            idf_dict[key] = math.log(float(doc_num) / float(idf_dict[key])) / math.log(10)
        vocab_idx_map = {}
        idx = 0
        for key in idf_dict.keys():
            vocab_idx_map[key] = idx
            idx += 1
        return vocab_idx_map, idf_dict

    def extract_dataset_instances(self, train_instances, train_file):
        """
        ../resources/data/sts-en-en/.. -> ../features/sts-en-en/..
        """
        self.feature_file = train_file.replace('resources/data/', 'features/')
        self.feature_file = self.feature_file.replace('.txt', '/' + self.feature_name + '.txt')

        return self.load_instances(train_instances)

    def extract(self, train_instance):
        pass


class DependencyGramFeature(Feature):
    def __init__(self, convey='count', **karwgs):
        super(DependencyGramFeature, self).__init__(**karwgs)
        self.convey = convey
        self.feature_name = self.feature_name + '-%s' % (convey)



    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            dep_sa, dep_sb = train_instance.get_dependency()
            dep_sa = [(dep[1], dep[2]) for dep in dep_sa]
            dep_sb = [(dep[1], dep[2]) for dep in dep_sb]
            seqs.append(dep_sa)
            seqs.append(dep_sb)
        self.idf_weight = utils.IDFCalculator(seqs)

    def extract(self, train_instance):
        dep_sa, dep_sb = train_instance.get_dependency()
        dep_sa = [(dep[1], dep[2]) for dep in dep_sa]
        dep_sb = [(dep[1], dep[2]) for dep in dep_sb]
        rs = [
            utils.sequence_match_features(dep_sa, dep_sb),
            utils.sequence_vector_features(dep_sa, dep_sb, self.idf_weight, convey=self.convey)
        ]
        features = [val for r in rs for val in r[0]]
        infos = [val for r in rs for val in r[1]]
        return features, infos


class DependencyRelationFeature(Feature):
    def __init__(self, convey='count', **karwgs):
        super(DependencyRelationFeature, self).__init__(**karwgs)
        self.convey = convey
        self.feature_name = self.feature_name + '-%s' % (convey)
        # self.feature_name = self.feature_name + '-%s' % (stopwords)

    def extract_information(self, train_instances):
        seqs = []
        for train_instance in train_instances:
            dep_sa, dep_sb = train_instance.get_dependency()
            seqs.append(dep_sa)
            seqs.append(dep_sb)
        self.idf_weight = utils.IDFCalculator(seqs)

    def extract(self, train_instance):
        dep_sa, dep_sb = train_instance.get_dependency()
        rs = [
            utils.sequence_match_features(dep_sa, dep_sb),
            utils.sequence_vector_features(dep_sa, dep_sb, self.idf_weight, convey=self.convey)
        ]
        features = [val for r in rs for val in r[0]]
        infos = [val for r in rs for val in r[1]]
        return features, infos
