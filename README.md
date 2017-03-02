# Semantic Textual Similarity Toolkits

## Installration
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
    $python test.py
    ```

## TODO
- chrome driver, to caputure mt features
- theano, lasagne, deep learning scores


## Embedding Feautures
- Pre-trained Embedding

## Machine Translation Features
- chromedriver 2.27 [download](https://chromedriver.storage.googleapis.com/index.html?path=2.27/)
- autoit