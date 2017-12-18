# A Word Aligner for English

This is a word aligner for English: given two English sentences, it aligns related words in the two sentences. It exploits the semantic and contextual similarities of the words to make alignment decisions.


## Ack
Initially, this is a fork of <i>[ma-sultan/monolingual-word-aligner](https://github.com/ma-sultan/monolingual-word-aligner)</i>, the aligner presented in [Sultan et al., 2015](https://github.com/FerreroJeremy/monolingual-word-aligner/blob/master/docs/DLS%40CU-%20Sentence%20Similarity%20from%20Word%20Alignment%20and%20Semantic%20Vector%20Composition.pdf) that has been very successful in [SemEval STS (Semantic Textual Similarity) Task](https://github.com/FerreroJeremy/monolingual-word-aligner/blob/master/docs/SemEval-2016%20Task%201-%20Semantic%20Textual%20Similarity%2C%20Monolingual%20and%20Cross-Lingual%20Evaluation.pdf) in recent years.


## Install
```bash
# download the repo

# require stopwords from nltk
python -m nltk.downloader stopwords

# require stanford corenlp
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip
unzip stanford-corenlp-full-2015-12-09.zip

# lanch the stanford CoreNLP
cd stanford-corenlp-full-2015-12-09/
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
# after this, you will find stanfordCoreNLP server at http://localhost:9000/

# python test_align.py
```

