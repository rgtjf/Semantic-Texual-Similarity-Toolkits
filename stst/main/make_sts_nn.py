# coding: utf8
from __future__ import print_function
import stst





if __name__ == '__main__':



    file_list = [config.STS_TRAIN_FILE, config.STS_DEV_FILE, config.STS_TEST_FILE]
    print('\n'.join(file_list))
    sentences, sentence_tags = get_all_instance(file_list)
    print(sentences[:5])
    print(sentence_tags[:5])

    idf_dict = utils.IDFCalculator(sentences)


    tagged_sentences = []
    for sentence, tag in zip(sentences, sentence_tags):
        tagged_sentences.append(TaggedDocument(words=sentence, tags=[tag]))

    doc2vec_model = Doc2Vec(tagged_sentences, size=25, window=3, min_count=0, workers=10, iter=1000)

    doc2vec_model.save(config.EX_DICT_DIR + '/doc2vec.model')

    # model.add(Doc2VecGlobalFeature())

    # model.make_feature_file()
