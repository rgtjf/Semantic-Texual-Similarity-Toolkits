import stst

# Define Model
gb = stst.Classifier(stst.GradientBoostingRegression())
model = stst.Model('S1-gb', gb)

# Add features to the Model
model.add(stst.BOWFeature(stopwords=False))
model.add(stst.BOWFeature(stopwords=True))

model.add(stst.BOWGlobalFeature(stopwords=False))
model.add(stst.BOWGlobalFeature(stopwords=True))


# train and test
train_file = './data/stsbenchmark/sts-train.csv'
dev_file  = './data/stsbenchmark/sts-dev.csv'
test_file = './data/stsbenchmark/sts-test.csv'

import tools
tools.make_sts_nn([train_file, dev_file, test_file])
# stst.dict_utils.DictCreater().create_global_idf([train_file, dev_file, test_file])

# init the server and input the address
nlp = stst.StanfordNLP('http://localhost:9000')

# parse data
train_instances = stst.load_parse_data(train_file, nlp)
dev_instances = stst.load_parse_data(dev_file, nlp)

# train and test
model.train(train_instances, train_file)
model.test(dev_instances, dev_file)

# evaluation
dev_pearsonr = stst.eval_output_file(model.output_file)
print('Dev:', dev_pearsonr)

# test on new data set
test_instances= stst.load_parse_data(test_file, nlp)
model.test(test_instances, test_file)
test_pearsonr = stst.eval_output_file(model.output_file)
print('Test:', test_pearsonr)

recod_file = './data/records.csv'
stst.record(recod_file, dev_pearsonr, test_pearsonr, model)
