# Semantic Textual Similarity Toolkits

## Usage
- download the repo
- run the corenlp server
- python test.py

## Requirements
- standford CoreNLP 3.6.0
    - (download)[http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip]
    - Usage
    ```
    $ cd stanford-corenlp-full-2015-12-09/
    $ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer')
    ```
- python
    - requirements.txt
    - Usage
    ```
    $ pip install -r requirements.txt
    ```

## TODO
- embedding features, run the embedding servefr
- chrome driver, to caputure mt features
- theano, lasagne, dl scores

## Usage
```python

train_file = './data/stsbenchmark/sts-train.csv'
dev_file  = './data/stsbenchmark/sts-dev.csv'
test_file = './data/stsbenchmark/sts-test.csv'

import stst

# Define Model
gb = stst.Classifier(stst.GradientBoostingRegression())
model = stst.Model('S1-gb', gb)

# Add features to the Model
model.add(stst.LexicalFeature())
model.add(stst.ShortSentenceFeature())

model.add(stst.nGramOverlapFeature(type='lemma'))
model.add(stst.nGramOverlapFeature(type='word'))
model.add(stst.nCharGramOverlapFeature(stopwords=True))
model.add(stst.nCharGramOverlapFeature(stopwords=False))

model.add(stst.nGramOverlapBeforeStopwordsFeature(type='lemma'))
model.add(stst.nGramOverlapBeforeStopwordsFeature(type='word'))

model.add(stst.WeightednGramMatchFeature(type='lemma'))
model.add(stst.WeightednGramMatchFeature(type='word'))

model.add(stst.BOWFeature(stopwords=False))

model.add(stst.AlignmentFeature())
model.add(stst.IdfAlignmentFeature())
model.add(stst.PosAlignmentFeature())

# model.add(stst.POSFeature())


# train and test

# init the server and input the address
nlp = stst.StanfordNLP('http://localhost:9000')

# parse data
train_instances = stst.load_parse_data(train_file, nlp)
dev_instances = stst.load_parse_data(dev_file, nlp)

# train and test
model.train(train_instances, train_file)
model.test(dev_instances, dev_file)

# evaluation
pearsonr = stst.eval_output_file(model.output_file)
print('Dev:', pearsonr)

# test on new data set
test_instances= stst.load_parse_data(test_file, nlp)
model.test(test_instances, test_file)
pearsonr = stst.eval_output_file(model.output_file)
print('Test:', pearsonr)
```