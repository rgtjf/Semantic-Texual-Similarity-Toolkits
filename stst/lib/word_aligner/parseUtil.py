# coding: utf8
from __future__ import print_function
from __future__ import unicode_literals

import json

##############################################################################################################################
def ner_word_annotator(parse_sent):
    """
    Generate NER Tag Sequence:
        - FORMAT: [[char_begin, char_end], word_index, word, ner] only ner != 'O'
    """
    res = []
    wordIndex = 1
    for i in range(len(parse_sent['sentences'][0]['tokens'])):
        tag = [
                [
                    parse_sent['sentences'][0]['tokens'][i]['characterOffsetBegin'],
                    parse_sent['sentences'][0]['tokens'][i]['characterOffsetEnd']
                ],
                wordIndex,
                parse_sent['sentences'][0]['tokens'][i]['word'],
                parse_sent['sentences'][0]['tokens'][i]['ner'],
            ]
        wordIndex += 1
        if tag[3] != 'O':
            res.append(tag)
    return res
##############################################################################################################################


##############################################################################################################################
def ner(parse_sent):
    """
    Generate Merged NER Sequence
        - FORMAT: [[ne], [char], [word]]
    Merge the same and adjacent NER tag into one list
    """

    nerWordAnnotations = ner_word_annotator(parse_sent)

    namedEntities = []
    currentNE = []
    currentCharacterOffsets = []
    currentWordOffsets = []

    for i in range(len(nerWordAnnotations)):

        # If the ner i == i-1, then do nothing
        # If i == 0, then do nothing
        if i == 0:
            pass
        elif i != 0 and nerWordAnnotations[i][3] == nerWordAnnotations[i - 1][3] \
                and nerWordAnnotations[i][1] == nerWordAnnotations[i - 1][1] + 1:
            pass
        else:
            namedEntities.append([currentCharacterOffsets, currentWordOffsets, currentNE,
                                  nerWordAnnotations[i - 1][3]])
            currentNE = []
            currentCharacterOffsets = []
            currentWordOffsets = []

        currentNE.append(nerWordAnnotations[i][2])
        currentCharacterOffsets.append(nerWordAnnotations[i][0])
        currentWordOffsets.append(nerWordAnnotations[i][1])

    if len(currentNE) != 0:
        ner_last_index = len(nerWordAnnotations) - 1
        namedEntities.append([currentCharacterOffsets, currentWordOffsets, currentNE,
                              nerWordAnnotations[ner_last_index][3]])

    return namedEntities
##############################################################################################################################




##############################################################################################################################
def posTag(parse_sent):
    """
    Generate POS Tag Sequence:
        - FORMAT: [[char_begin, char_end], word_index, word, pos]
    """
    res = []
    wordIndex = 1
    for i in range(len(parse_sent['sentences'][0]['tokens'])):
        tag = [
                [
                    parse_sent['sentences'][0]['tokens'][i]['characterOffsetBegin'],
                    parse_sent['sentences'][0]['tokens'][i]['characterOffsetEnd']
                ],
                wordIndex,
                parse_sent['sentences'][0]['tokens'][i]['word'],
                parse_sent['sentences'][0]['tokens'][i]['pos']
        ]
        wordIndex += 1
        res.append(tag)
    return res
##############################################################################################################################




##############################################################################################################################
def lemmatize(parseResult):
    """
    Generate Lemma Sequence:
        - FORMAT: [[char_begin, char_end], word_index, word, lemma]
    """
    res = []
    wordIndex = 1
    for i in range(len(parseResult['sentences'][0]['tokens'])):
        tag = [
                [
                    parseResult['sentences'][0]['tokens'][i]['characterOffsetBegin'],
                    parseResult['sentences'][0]['tokens'][i]['characterOffsetEnd']
                ],
                wordIndex,
                parseResult['sentences'][0]['tokens'][i]['word'],
                parseResult['sentences'][0]['tokens'][i]['lemma']
            ]
        wordIndex += 1
        res.append(tag)

    return res
##############################################################################################################################




##############################################################################################################################
def dependencyParseAndPutOffsets(parseResult):
    """
    returns dependency parse of the sentence where each item is of the form:
    (rel, left{charStartOffset charEndOffset wordIndex}, right{charStartOffset, charEndOffset, wordIndex})
    """

    dParse = parseResult['sentences'][0]['basic-dependencies']
    tokens = parseResult['sentences'][0]['tokens']

    result = []
    for item in dParse:
        rel = item['dep']
        leftIndex = item['governor']
        rightIndex = item['dependent']

        if 'governorGloss' not in item:
            if leftIndex == 0:
                item['governorGloss'] = 'ROOT'
            else:
                item['governorGloss'] = tokens[leftIndex- 1]['word']

        if 'dependentGloss' not in item:
            if rightIndex == 0:
                item['dependentGloss'] = 'ROOT'
            else:
                item['dependentGloss'] = tokens[rightIndex - 1]['word']

        # left and right order is important
        left = item['governorGloss']
        right = item['dependentGloss']

        leftIndex -= 1
        rightIndex -= 1
        left += '{{{0} {1} {2}}}'.format(tokens[leftIndex]['characterOffsetBegin'],
                                       tokens[leftIndex]['characterOffsetEnd'],
                                       leftIndex + 1)

        right += '{{{0} {1} {2}}}'.format(tokens[rightIndex]['characterOffsetBegin'],
                                        tokens[rightIndex]['characterOffsetEnd'],
                                        rightIndex + 1)
        result.append([rel, left, right])

    return result

##############################################################################################################################



##############################################################################################################################
def findParents(dependencyParse, wordIndex, word):
    # word index assumed to be starting at 1
    # the third parameter is needed because of the collapsed representation of the dependencies...

    tokensWithIndices = ((int(item[2].split('{')[1].split('}')[0].split(' ')[2]), item[2].split('{')[0]) for item in
                         dependencyParse)
    tokensWithIndices = list(set(tokensWithIndices))
    tokensWithIndices = sorted(tokensWithIndices, key=lambda item: item[0])

    wordIndexPresentInTheList = False
    for item in tokensWithIndices:
        if item[0] == wordIndex:
            wordIndexPresentInTheList = True
            break

    parentsWithRelation = []

    if wordIndexPresentInTheList:
        for item in dependencyParse:
            currentIndex = int(item[2].split('{')[1].split('}')[0].split(' ')[2])
            if currentIndex == wordIndex:
                parentsWithRelation.append(
                    [int(item[1].split('{')[1].split('}')[0].split(' ')[2]), item[1].split('{')[0], item[0]])
    else:
        # find the closest following word index which is in the list
        nextIndex = 0
        for i in range(len(tokensWithIndices)):
            if tokensWithIndices[i][0] > wordIndex:
                nextIndex = tokensWithIndices[i][0]
                break
        if nextIndex == 0:
            return []  # ?
        for i in range(len(dependencyParse)):
            if int(dependencyParse[i][2].split('{')[1].split('}')[0].split(' ')[2]) == nextIndex:
                pos = i
                break
        for i in range(pos, len(dependencyParse)):
            if '_' in dependencyParse[i][0] and word in dependencyParse[i][0]:
                parent = [int(dependencyParse[i][1].split('{')[1].split('}')[0].split(' ')[2]),
                          dependencyParse[i][1].split('{')[0], dependencyParse[i][0]]
                parentsWithRelation.append(parent)
                break

    return parentsWithRelation

##############################################################################################################################




##############################################################################################################################
def findChildren(dependencyParse, wordIndex, word):
    # word index assumed to be starting at 1
    # the third parameter is needed because of the collapsed representation of the dependencies...

    tokensWithIndices = ((int(item[2].split('{')[1].split('}')[0].split(' ')[2]), item[2].split('{')[0]) for item in
                         dependencyParse)
    tokensWithIndices = list(set(tokensWithIndices))
    tokensWithIndices = sorted(tokensWithIndices, key=lambda item: item[0])

    wordIndexPresentInTheList = False
    for item in tokensWithIndices:
        if item[0] == wordIndex:
            wordIndexPresentInTheList = True
            break

    childrenWithRelation = []

    if wordIndexPresentInTheList:
        for item in dependencyParse:
            currentIndex = int(item[1].split('{')[1].split('}')[0].split(' ')[2])
            if currentIndex == wordIndex:
                childrenWithRelation.append(
                    [int(item[2].split('{')[1].split('}')[0].split(' ')[2]), item[2].split('{')[0], item[0]])
    else:
        # find the closest following word index which is in the list
        nextIndex = 0
        for i in range(len(tokensWithIndices)):
            if tokensWithIndices[i][0] > wordIndex:
                nextIndex = tokensWithIndices[i][0]
                break

        if nextIndex == 0:
            return []
        for i in range(len(dependencyParse)):
            if int(dependencyParse[i][2].split('{')[1].split('}')[0].split(' ')[2]) == nextIndex:
                pos = i
                break
        for i in range(pos, len(dependencyParse)):
            if '_' in dependencyParse[i][0] and word in dependencyParse[i][0]:
                child = [int(dependencyParse[i][2].split('{')[1].split('}')[0].split(' ')[2]),
                         dependencyParse[i][2].split('{')[0], dependencyParse[i][0]]
                childrenWithRelation.append(child)
                break

    return childrenWithRelation

##############################################################################################################################



if __name__ == '__main__':

    parse_sent = u"""[{"sentences": [{"tokens": [{"index": 1, "word": "Waba", "lemma": "waba", "after": " ", "pos": "NN", "characterOffsetEnd": 4, "characterOffsetBegin": 0, "originalText": "Waba", "ner": "PERSON", "before": ""}, {"index": 2, "word": "emerges", "lemma": "emerge", "after": " ", "pos": "VBZ", "characterOffsetEnd": 12, "characterOffsetBegin": 5, "originalText": "emerges", "ner": "O", "before": " "}, {"index": 3, "word": "new", "lemma": "new", "after": " ", "pos": "JJ", "characterOffsetEnd": 16, "characterOffsetBegin": 13, "originalText": "new", "ner": "O", "before": " "}, {"index": 4, "word": "NLC", "lemma": "nlc", "after": " ", "pos": "NN", "characterOffsetEnd": 20, "characterOffsetBegin": 17, "originalText": "NLC", "ner": "ORGANIZATION", "before": " "}, {"index": 5, "word": "president", "lemma": "president", "after": "", "pos": "NN", "characterOffsetEnd": 30, "characterOffsetBegin": 21, "originalText": "president", "ner": "O", "before": " "}], "index": 0, "basic-dependencies": [{"dep": "ROOT", "dependent": 2, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "emerges"}, {"dep": "nsubj", "dependent": 1, "governorGloss": "emerges", "governor": 2, "dependentGloss": "Waba"}, {"dep": "amod", "dependent": 3, "governorGloss": "president", "governor": 5, "dependentGloss": "new"}, {"dep": "compound", "dependent": 4, "governorGloss": "president", "governor": 5, "dependentGloss": "NLC"}, {"dep": "dobj", "dependent": 5, "governorGloss": "emerges", "governor": 2, "dependentGloss": "president"}], "parse": "(ROOT\n  (S\n    (NP (NN Waba))\n    (VP (VBZ emerges)\n      (NP (JJ new) (NN NLC) (NN president)))))", "collapsed-dependencies": [{"dep": "ROOT", "dependent": 2, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "emerges"}, {"dep": "nsubj", "dependent": 1, "governorGloss": "emerges", "governor": 2, "dependentGloss": "Waba"}, {"dep": "amod", "dependent": 3, "governorGloss": "president", "governor": 5, "dependentGloss": "new"}, {"dep": "compound", "dependent": 4, "governorGloss": "president", "governor": 5, "dependentGloss": "NLC"}, {"dep": "dobj", "dependent": 5, "governorGloss": "emerges", "governor": 2, "dependentGloss": "president"}], "collapsed-ccprocessed-dependencies": [{"dep": "ROOT", "dependent": 2, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "emerges"}, {"dep": "nsubj", "dependent": 1, "governorGloss": "emerges", "governor": 2, "dependentGloss": "Waba"}, {"dep": "amod", "dependent": 3, "governorGloss": "president", "governor": 5, "dependentGloss": "new"}, {"dep": "compound", "dependent": 4, "governorGloss": "president", "governor": 5, "dependentGloss": "NLC"}, {"dep": "dobj", "dependent": 5, "governorGloss": "emerges", "governor": 2, "dependentGloss": "president"}]}]}, {"sentences": [{"tokens": [{"index": 1, "word": "Waiting", "lemma": "wait", "after": " ", "pos": "VBG", "characterOffsetEnd": 7, "characterOffsetBegin": 0, "originalText": "Waiting", "ner": "O", "before": ""}, {"index": 2, "word": "for", "lemma": "for", "after": " ", "pos": "IN", "characterOffsetEnd": 11, "characterOffsetBegin": 8, "originalText": "for", "ner": "O", "before": " "}, {"index": 3, "word": "the", "lemma": "the", "after": " ", "pos": "DT", "characterOffsetEnd": 15, "characterOffsetBegin": 12, "originalText": "the", "ner": "O", "before": " "}, {"index": 4, "word": "next", "lemma": "next", "after": " ", "pos": "JJ", "characterOffsetEnd": 20, "characterOffsetBegin": 16, "originalText": "next", "ner": "O", "before": " "}, {"index": 5, "word": "president", "lemma": "president", "after": "", "pos": "NN", "characterOffsetEnd": 30, "characterOffsetBegin": 21, "originalText": "president", "ner": "O", "before": " "}], "index": 0, "basic-dependencies": [{"dep": "ROOT", "dependent": 1, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "Waiting"}, {"dep": "case", "dependent": 2, "governorGloss": "president", "governor": 5, "dependentGloss": "for"}, {"dep": "det", "dependent": 3, "governorGloss": "president", "governor": 5, "dependentGloss": "the"}, {"dep": "amod", "dependent": 4, "governorGloss": "president", "governor": 5, "dependentGloss": "next"}, {"dep": "nmod", "dependent": 5, "governorGloss": "Waiting", "governor": 1, "dependentGloss": "president"}], "parse": "(ROOT\n  (S\n    (VP (VBG Waiting)\n      (PP (IN for)\n        (NP (DT the) (JJ next) (NN president))))))", "collapsed-dependencies": [{"dep": "ROOT", "dependent": 1, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "Waiting"}, {"dep": "case", "dependent": 2, "governorGloss": "president", "governor": 5, "dependentGloss": "for"}, {"dep": "det", "dependent": 3, "governorGloss": "president", "governor": 5, "dependentGloss": "the"}, {"dep": "amod", "dependent": 4, "governorGloss": "president", "governor": 5, "dependentGloss": "next"}, {"dep": "nmod:for", "dependent": 5, "governorGloss": "Waiting", "governor": 1, "dependentGloss": "president"}], "collapsed-ccprocessed-dependencies": [{"dep": "ROOT", "dependent": 1, "governorGloss": "ROOT", "governor": 0, "dependentGloss": "Waiting"}, {"dep": "case", "dependent": 2, "governorGloss": "president", "governor": 5, "dependentGloss": "for"}, {"dep": "det", "dependent": 3, "governorGloss": "president", "governor": 5, "dependentGloss": "the"}, {"dep": "amod", "dependent": 4, "governorGloss": "president", "governor": 5, "dependentGloss": "next"}, {"dep": "nmod:for", "dependent": 5, "governorGloss": "Waiting", "governor": 1, "dependentGloss": "president"}]}]}, 1.0]"""

    parse_json = json.loads(parse_sent, strict=False)
    sa, sb, score = parse_json

    print(sa)
    print(ner_word_annotator(sa))
    print(ner(sa))
    print(posTag(sa))
    print(dependencyParseAndPutOffsets(sa))
    print(findParents(dependencyParseAndPutOffsets(sa), 1, 'Weba'))
    print(findChildren(dependencyParseAndPutOffsets(sa), 2, 'emerges'))

    # print(ner(json.loads(parse_sent)))