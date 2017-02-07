
def test_baidu_api():
    import stst.translator.baidu_api as baidu_api
    print(baidu_api.translator.translate('Qué tipo de pastel es este?'))


def test_ms_api():
    import stst.translator.mstranslator_api as ms_api
    print(ms_api.translator.translate('Qué tipo de pastel es este?'))


def test_google_api():
    import stst.translator.google_api as google_api
    print(google_api.translator.translate('Qué tipo de pastel es este?'))
