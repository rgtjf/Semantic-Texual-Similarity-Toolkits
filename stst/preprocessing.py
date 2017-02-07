import logging, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import string
import sys

reload(sys)
sys.setdefaultencoding('utf8')

puncts = string.punctuation + '\r\n``\'\'..'

log_format = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(format=log_format, datefmt='%Y-%m-%d %H:%M:%S %p',
                    filename='preprocessing.log', filemode='w',
                    level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(log_format)
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

wl = WordNetLemmatizer()
fp = open('english.stopwords.txt', 'r')
stopwords = [word.strip() for word in fp.readlines()]
logging.info('stopwords:\n%s' % stopwords)
fp.close()


# input format:(word, pos)
# output format: set of word
def get_synonyms(word_with_pos):
    if word_with_pos[1] == '#':
        return []
    synsets = wn.synsets(word_with_pos[0], word_with_pos[1])
    synonyms = []
    for synset in synsets:
        name = synset.name().split('.')
        synonym = (name[0], name[1])
        if not synonym == word_with_pos:
            synonyms.append(synonym)
    return set([word for word, pos in synonyms])


def posTransform(pos):
    if pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS':
        pos = 'n'
    elif pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == "VBP" or pos == "VBZ":
        pos = 'v'
    elif pos == 'JJ' or pos == 'JJR' or pos == 'JJS':
        pos = 'a'
    elif pos == 'RB' or pos == 'RBR' or pos == 'RBS':
        pos = 'r'
    else:
        pos = '#'
    return pos


def loadSlang(file, sep=' '):
    fp = open(file, 'r')
    slang_dict = {}
    for line in fp.readlines():
        fields = line.strip().split(sep)
        slang_dict[fields[0]] = fields[1]
    fp.close()
    logging.info('slang dict loaded! #=%d' % (len(slang_dict)))
    return slang_dict


def elongatedWord(word):
    new_word = ''
    pre_ch = ''
    cnt = 0
    for ch in word:
        if ch == pre_ch:
            cnt += 1
        else:
            for i in range(cnt if cnt <= 2 else 2):
                new_word += pre_ch
            cnt = 1
            pre_ch = ch
    for i in range(cnt if cnt <= 2 else 2):
        new_word += pre_ch
    return new_word


def transform(sent):
	text = nltk.word_tokenize(sent)
	taggedSen = nltk.pos_tag(text)
	items = taggedSen
	sent = []
	for item in items:
		fields = item
		word = fields[0]
		pos = posTransform(fields[1])
		for w in word.split(' '):
			lemma = wl.lemmatize(w, 'n' if pos == '#' else pos)
			e_lemma = elongatedWord(lemma)
			if e_lemma != lemma:
				logging.info('%s->%s'%(lemma, e_lemma))
				lemma = e_lemma
			if lemma in puncts:
				continue
			if lemma == '\'s':
				lemma = 'be'
			if lemma == '\'re':
				lemma = 'be'
			if lemma == '\'ve':
				lemma = 'have'
			if lemma == '\'d':
				lemma = 'would'
			if lemma == '\'m':
				lemma = 'be'
			if lemma == '\'t':
				lemma = 'not'
			sent.append([lemma, pos, False])
	logging.debug(sent)
	return sent


def algin(sent1, sent2):
    for idx1 in range(len(sent1)):
        if sent1[idx1][2]:
            continue
        for idx2 in range(len(sent2)):
            if sent1[idx1][0] == sent2[idx2][0]:
                sent1[idx1][2] = True
                sent2[idx2][2] = True
                break
            else:
                synonyms = get_synonyms((sent2[idx2][0], sent2[idx2][1]))
                if sent1[idx1][0] in synonyms:
                    logging.info('synonym: %s %s' % (sent1[idx1][0], sent2[idx2][0]))
                    sent1[idx1][2] = True
                    sent2[idx2][2] = True
                    sent1[idx1][0] = sent2[idx2][0]
    return sent1, sent2
