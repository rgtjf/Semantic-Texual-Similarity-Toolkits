from stst.lib.pycorenlp.corenlp import StanfordCoreNLP

class StanfordNLP:
    def __init__(self, server_url='http://localhost:9000'):
        self.server = StanfordCoreNLP(server_url)

    def parse(self, text):
        output = self.server.annotate(text, properties={
            'timeout': '50000',
            'ssplit.isOneSentence': 'true',
            'depparse.DependencyParseAnnotator': 'basic',
            'annotators': 'tokenize,lemma,ssplit,pos,depparse,parse,ner',
            # 'annotators': 'tokenize,lemma,ssplit,pos,ner',
            'outputFormat': 'json'
        })

        return output

# nlp = StanfordNLP()
