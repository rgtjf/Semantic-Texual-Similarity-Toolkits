mkdir data
cd data

# download the STSBenchmark Dataset
wget http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz
tar zxvf Stsbenchmark.tar.gz
rm -rf Stsbenchmark.tar.gz

# download the stanford CoreNLP 3.6.0
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip
unzip stanford-corenlp-full-2015-12-09.zip

# lanch the stanford CoreNLP
cd stanford-corenlp-full-2015-12-09/
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
# after this, you will find stanfordCoreNLP server at http://localhost:9000/

# install requirements
cd ..
pip install -r requirements.txt

# install nltk stopwords
python -m nltk.downloader stopwords