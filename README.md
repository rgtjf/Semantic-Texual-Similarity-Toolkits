# Semantic Textual Similarity Toolkits

A light version of STST

Our goal:
1. easy and fast to build a nlp system
2. it's easy to use a library to be integrated into other systems.
3. and so on.



## How to use it?
```python
import stst

classifier = stst.Classifier('classifier')

model = stst.Model('model', classifier)

class DefineFeature(stst.Feature):
    def extract(self):
        pass

class DefineData(stst.Data):

    def __init__(self):
        pass

    def load_from_file():
        pass

    def __iter__(self):
        pass

class DefineEval(stst.Eval):

    def measure(self, a, b):
        pass


model.add(Define(Feature))
train_data = {文本/数字} field


model.train_model(train_data)
model.test_model(test_data)


```


## Installation
1. download the repo
2. run the corenlp server
   standford CoreNLP 3.6.0 [download](http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip)
    - Usage
    ```
    $ cd stanford-corenlp-full-2015-12-09/
    $ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer')
    ```
3. python
    - requirements.txt
    - Usage
    ```
    $ pip install -r requirements.txt
    ```
    - Download the NLTK stopword corpus:
    ```
    $ python -m nltk.downloader stopwords
    ```
4. Run

    ```
    $python example.py
    ```

## TODO
- chrome driver, to caputure mt features
- theano, lasagne, deep learning scores
   - stst/main/make_sts_nn.py
   - stst/main/make_sts_iclr.py


## Embedding Feautures
- Pre-trained Embedding

## Machine Translation Features
- chromedriver 2.27 [download](https://chromedriver.storage.googleapis.com/index.html?path=2.27/)
- autoit
