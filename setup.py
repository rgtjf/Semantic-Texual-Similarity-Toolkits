from distutils.core import setup
from setuptools import find_packages

setup(
    name='stst',
    packages=find_packages(),
    version='0.2.0',
    description='stst',
    author='rgtjf',
    author_email='rgtjf1@163.com',
    url='https://github.com/rgtjf/Semantic-Texual-Similarity-Toolkits',
    download_url='https://github.com/rgtjf/Semantic-Texual-Similarity-Toolkits',
    license='MIT',
    keywords=['natural language processing', 'semantic texual similarity', 'stst'],  # arbitrary keywords
    classifiers=[
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],
    install_requires = ['pyprind', 'python-jsonrpc', 'gensim', 'selenium']
)
