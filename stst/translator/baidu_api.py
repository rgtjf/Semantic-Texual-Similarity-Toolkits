import requests
import random
import hashlib


class Translator(object):

    """
    Ref: http://api.fanyi.baidu.com/api/trans/product/apidoc
    """

    def __init__(self, app_id, secret_key):

        self.app_id = app_id
        self.secret_key = secret_key
        self.url = "http://api.fanyi.baidu.com/api/trans/vip/translate"

    def _generate_sign(self, params):
        sign = params['appid'] + params['q'] + str(params['salt']) + params['secret_key']
        m = hashlib.md5()
        m.update(sign.encode('utf8'))
        sign = m.hexdigest()
        params['sign'] = sign
        return params

    def translate(self, text, from_lang='auto', to_lang="en", only_one_sentence=True):
        try:
            p = {
                "appid": self.app_id,
                "secret_key": self.secret_key,
                "q": text,
                "from": from_lang,
                "salt": random.randint(32768, 65536),
                "to": to_lang
            }
            p = self._generate_sign(p)
            r = requests.get(self.url, params=p)
            data = r.json()
            if only_one_sentence:
                trans_text = data['trans_result'][0]['dst']
            else:
                raise NotImplementedError('only_one_sentence')

        except Exception:
            trans_text = 'ExceptionError'

        return trans_text


translator = Translator('20161026000030818', 'grCst5Xx9WgyvuaeKW0P')