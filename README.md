# Semantic Textual Similarity Toolkits

## Usage
- download the repo
- run the corenlp server
- python test.py

## Requirements
- standford CoreNLP 3.6.0
    - [download](http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip)
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
    - Download the NLTK stopword corpus:
    ```
    $ python -m nltk.downloader stopwords
    ```
## TODO
- chrome driver, to caputure mt features
- theano, lasagne, dl scores

## Usage
```python
$ python test.py
```