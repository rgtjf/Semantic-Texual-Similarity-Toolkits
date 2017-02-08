
train_file = '../data/sts-en-en/input.all.txt'
dev_file = '../data/sts-en-en/manual/STS2016.manual.answer-answer.txt'
dev_gs_file = '../data/sts-en-en/STS2016.gs.answer-answer.txt'

TEST_FILES = {
    'sts-en-en': ['STS2016.input.answer-answer.txt', 'STS2016.input.headlines.txt', 'STS2016.input.plagiarism.txt',
                  'STS2016.input.postediting.txt', 'STS2016.input.question-question.txt']
}

import stst

gb = stst.Classifier(stst.GradientBoostingRegression())
model = stst.Model('S1-gb', gb)

model.add(stst.LexicalFeature())


train_instances = stst.load_parse_data(train_file, flag=True)
dev_instances = stst.load_parse_data(dev_file, flag=True)
model.train(train_instances, train_file)
model.test(dev_instances, dev_file)
pearsonr = stst.eval_file(model.output_file, dev_gs_file)