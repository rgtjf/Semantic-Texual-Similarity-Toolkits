
train_file = '../data/sts-en-en/input.all.txt'
dev_file = '../data/sts-en-en/manual/STS2016.manual.answer-answer.txt'
dev_gs_file = '../data/sts-en-en/STS2016.gs.answer-answer.txt'

test_file = '../data/sts-en-en/manual/STS2016.manual.headlines.txt'



TEST_FILES = {
    'sts-en-en': ['STS2016.input.answer-answer.txt', 'STS2016.input.headlines.txt', 'STS2016.input.plagiarism.txt',
                  'STS2016.input.postediting.txt', 'STS2016.input.question-question.txt']
}

import stst


gb = stst.Classifier(stst.GradientBoostingRegression())
model = stst.Model('S1-gb', gb)

model.add(stst.LexicalFeature())
model.add(stst.ShortSentenceFeature())
model.add(stst.nLemmaGramOverlapFeature())
model.add(stst.nWordGramOverlapFeature())
model.add(stst.nCharGramOverlapFeature())
model.add(stst.nCharGramNonStopwordsOverlapFeature())
model.add(stst.nWordGramOverlapBeforeStopwordsFeature())
model.add(stst.nLemmaGramOverlapBeforeStopwordsFeature())
model.add(stst.nLemmaGramOverlapMatchFeature())
# model.add(stst.nLemmaGramOverlapBeforeStopwordsMatchFeature())
model.add(stst.BOWFeature(stopwords=False))
model.add(stst.AlignmentFeature())
model.add(stst.IdfAlignmentFeature())
model.add(stst.PosAlignmentFeature())

# model.add(stst.POSFeature())

train_instances = stst.load_parse_data(train_file, flag=False)
dev_instances = stst.load_parse_data(dev_file, flag=False)
model.train(train_instances, train_file)
model.test(dev_instances, dev_file)
pearsonr = stst.eval_file(model.output_file, dev_gs_file)
print(pearsonr)

test_instances= stst.load_parse_data(dev_file, server_url='http://49.52.10.39:9000')
model.test(test_instances, test_file)