# coding: utf8
import json
import requests
import traceback
import six


class StanfordCoreNLP:
    def __init__(self, server_url):
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url

    def annotate(self, text, properties=None):
        if isinstance(text, six.text_type):
            text = text.encode('utf8')
        assert isinstance(text, bytes)
        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)

        # Checks that the Stanford CoreNLP server is started.
        try:
            requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the CoreNLP server e.g.\n'
                            '$ cd stanford-corenlp-full-2015-12-09/ \n'
                            '$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer')

        data = text  # text in instance utf8, text.encode('utf8')

        r = requests.post(
            self.server_url, params={
                'properties': str(properties)
            }, data=data, headers={'Connection': 'close'})
        output = r.text
        if ('outputFormat' in properties
            and properties['outputFormat'] == 'json'):
            try:
                output = json.loads(output, encoding='utf8', strict=True)
            except Exception:
                # print(e)
                try:
                    output = json.loads(output, encoding='utf8', strict=False)
                except Exception:
                    traceback.print_exc()
                    pass
        return output


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

nlp = StanfordNLP()


def stanford_tokenize(s, type='word'):
    """
    Tokenization of the given text using StanfordCoreNLP
    Args:
        s: text
        type: 'word'/'lemma'
    Returns:
        list of tokens
    """
    parsetext = nlp.parse(s)
    tokens = parsetext['sentences'][0]['tokens']
    result = []
    for token in tokens:
        result.append(token[type])
    return result


if __name__ == '__main__':

    parsetext = nlp.parse(u'I love China.')
    print(json.dumps(parsetext, indent=2))

    print(stanford_tokenize('She loves China.', type='lemma'))