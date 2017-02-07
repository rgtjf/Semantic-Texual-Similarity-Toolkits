"""
Ref: 1. https://www.microsoft.com/en-us/translator/getstarted.aspx
     2. https://www.microsoft.com/en-us/translator/translatorapi.aspx
     3. https://msdn.microsoft.com/en-us/library/dd576287.aspx
     4. https://msdn.microsoft.com/en-us/library/hh454950.aspx
Getting started using the Translator service is a snap to scale to your needs.
Sign up for a free subscription for volumes up to 2 million characters per month,
purchase a monthly subscription for higher volumes from the Windows Azure Marketplace,
or through volume licensing for Enterprise customers. Get started today.
"""
import json
import requests
import urllib.request, urllib.parse, urllib.error
from xml.etree import ElementTree



class Translator(object):
    """Microsoft Translator API
    Ref: https://github.com/MicrosoftTranslator/PythonConsole/blob/master/MTPythonSampleCode/MTPythonSampleCode.py
    Usage:
        ```
        from mstranslator_api import Translator
        translator = Translator('ENTER YOU CLIENT ID', 'ENTER YOUR CLIENT SECRET')
        translator.translate(text, lang_to=lang_to)
        ```

    """
    def __init__(self, client_id, client_secret):

        self.finalToken = self.GetToken(client_id, client_secret)


    def GetToken(self, client_id, client_secret): #Get the access token from ADM, token is good for 10 minutes
        urlArgs = {
            'client_id': client_id,
            'client_secret': client_secret,
            'scope': 'http://api.microsofttranslator.com',
            'grant_type': 'client_credentials'
        }

        oauthUrl = 'https://datamarket.accesscontrol.windows.net/v2/OAuth2-13'

        try:
            oauthToken = json.loads(requests.post(oauthUrl, data = urllib.parse.urlencode(urlArgs)).content.decode('utf-8')) #make call to get ADM token and parse json
            finalToken = "Bearer " + oauthToken['access_token'] #prepare the token
        except OSError:
            pass

        return finalToken
    #End GetToken

    def translate(self, text, lang_to='en', max_iter_errors=3):
        # Call to Microsoft Translator Service

        headers = {"Authorization ": self.finalToken}
        translateUrl = "http://api.microsofttranslator.com/v2/Http.svc/Translate?text={}&to={}".format(text, lang_to)

        while max_iter_errors >= 0:

            try:
                translationData = requests.get(translateUrl, headers=headers)  # make request
                translation = ElementTree.fromstring(translationData.text.encode('utf-8'))  # parse xml return values
                translation_text = translation.text     # display translation
                break

            except OSError:
                translation_text = 'OSError'

            max_iter_errors = max_iter_errors - 1
        return translation_text




translator = Translator('ecnu_translator', 'sc6x4pJQDBtyqmGhhRmOknjeeXcL1YGBUTkvZLtn62k=')
