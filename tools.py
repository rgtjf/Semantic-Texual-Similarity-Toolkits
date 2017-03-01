from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import stst

def make_sts_nn(file_list):
    sentences, sentence_tags = stst.load_sentences(file_list)

    tagged_sentences = []
    for sentence, tag in zip(sentences, sentence_tags):
        tagged_sentences.append(TaggedDocument(words=sentence, tags=[tag]))

    print('Train doc2vec ...')

    doc2vec_model = Doc2Vec(tagged_sentences, size=25, window=3, min_count=0, workers=10, iter=1000)

    doc2vec_model.save(stst.config.DICT_DIR + '/doc2vec.model')



def make_sts_iclr(train_file, ):
    pass

