# A Word Aligner for English

This is a word aligner for English: given two English sentences, it aligns related words in the two sentences. It exploits the semantic and contextual similarities of the words to make alignment decisions.

## Requirements

1) Python **NLTK**  
2) The [Python wrapper for Stanford CoreNLP](https://github.com/dasmith/stanford-corenlp-python)  

## Installation and Usage

1) Install the above tools.  
2) Change line 100 of corenlp.py, from "rel, left, right = map(lambda x: remove_id(x), split_entry)" to "rel, left, right = split_entry".  
3) Download the NLTK stopword corpus:  

	python -m nltk.downloader stopwords
4) Install jsonrpclib:  

	sudo pip install jsonrpclib
5) Download the aligner:  

	  git clone https://github.com/ma-sultan/monolingual-word-aligner.git  
6) Run the corenlp.py script to launch the server:  

	  python corenlp.py  
7) To view the aligner in action, run **testAlign.py**. (Word indexing starts at 1.)
