# coding: utf8
from __future__ import print_function

from stst.lib.pycorenlp.corenlp import StanfordCoreNLP
import json


class StanfordNLP:
    def __init__(self):
        self.server = StanfordCoreNLP('http://localhost:9000')

    def parse(self, text):
        output = self.server.annotate(text, properties={
            'ssplit.isOneSentence': 'true',
            'annotators': 'tokenize,lemma,ssplit,pos,depparse,parse,ner',
            # 'annotators': 'tokenize,lemma,ssplit,pos,ner',
            'outputFormat': 'json'
        })

        return output


##############################################################################################################################
def parseText(sentences):
    parseResult = nlp.parse(sentences)

    ### Request Time Out May Happen, it is so sad
    if isinstance(parseResult, str):
        # print parseResult
        print(parseResult)
        parseResult = {}

    return parseResult


##############################################################################################################################



##############################################################################################################################
def nerWordAnnotator(parseResult):
    res = []
    wordIndex = 1
    for i in range(len(parseResult['sentences'][0]['tokens'])):
        tag = [[
            parseResult['sentences'][0]['tokens'][i]['characterOffsetBegin'],
            parseResult['sentences'][0]['tokens'][i]['characterOffsetEnd']
        ],
            wordIndex,
            parseResult['sentences'][0]['tokens'][i]['word'], parseResult['sentences'][0]['tokens'][i]['ner']]
        wordIndex += 1
        if tag[3] != 'O':
            res.append(tag)
    return res


##############################################################################################################################


##############################################################################################################################
def ner(parseResult):
    nerWordAnnotations = nerWordAnnotator(parseResult)

    namedEntities = []
    currentNE = []
    currentCharacterOffsets = []
    currentWordOffsets = []

    for i in range(len(nerWordAnnotations)):

        if i == 0:
            currentNE.append(nerWordAnnotations[i][2])
            currentCharacterOffsets.append(nerWordAnnotations[i][0])
            currentWordOffsets.append(nerWordAnnotations[i][1])
            if len(nerWordAnnotations) == 1:
                namedEntities.append(
                    [currentCharacterOffsets, currentWordOffsets, currentNE, nerWordAnnotations[i - 1][3]])
                break
            continue

        if nerWordAnnotations[i][3] == nerWordAnnotations[i - 1][3] and nerWordAnnotations[i][1] == \
                        nerWordAnnotations[i - 1][1] + 1:
            currentNE.append(nerWordAnnotations[i][2])
            currentCharacterOffsets.append(nerWordAnnotations[i][0])
            currentWordOffsets.append(nerWordAnnotations[i][1])
            if i == len(nerWordAnnotations) - 1:
                namedEntities.append([currentCharacterOffsets, currentWordOffsets, currentNE, nerWordAnnotations[i][3]])
        else:
            namedEntities.append([currentCharacterOffsets, currentWordOffsets, currentNE, nerWordAnnotations[i - 1][3]])
            currentNE = [nerWordAnnotations[i][2]]
            currentCharacterOffsets = []
            currentCharacterOffsets.append(nerWordAnnotations[i][0])
            currentWordOffsets = []
            currentWordOffsets.append(nerWordAnnotations[i][1])
            if i == len(nerWordAnnotations) - 1:
                namedEntities.append([currentCharacterOffsets, currentWordOffsets, currentNE, nerWordAnnotations[i][3]])

    # print('namedeEntities', namedEntities)
    return namedEntities


##############################################################################################################################


##############################################################################################################################
def posTag(parseResult):
    res = []

    wordIndex = 1
    for i in range(len(parseResult['sentences'][0]['tokens'])):
        tag = [[parseResult['sentences'][0]['tokens'][i]['characterOffsetBegin'],
                parseResult['sentences'][0]['tokens'][i]['characterOffsetEnd']], wordIndex,
               parseResult['sentences'][0]['tokens'][i]['word'], parseResult['sentences'][0]['tokens'][i]['pos']]
        wordIndex += 1
        res.append(tag)

    return res


##############################################################################################################################




##############################################################################################################################
def lemmatize(parseResult):
    res = []

    wordIndex = 1
    for i in range(len(parseResult['sentences'][0]['tokens'])):
        tag = [[parseResult['sentences'][0]['tokens'][i]['characterOffsetBegin'],
                parseResult['sentences'][0]['tokens'][i]['characterOffsetEnd']], wordIndex,
               parseResult['sentences'][0]['tokens'][i]['word'], parseResult['sentences'][0]['tokens'][i]['lemma']]
        wordIndex += 1
        res.append(tag)

    return res


##############################################################################################################################





##############################################################################################################################
def dependencyParseAndPutOffsets(parseResult):
    # returns dependency parse of the sentence whhere each item is of the form (rel, left{charStartOffset, charEndOffset, wordNumber}, right{charStartOffset, charEndOffset, wordNumber})

    dParse = parseResult['sentences'][0]['basic-dependencies']
    tokens = parseResult['sentences'][0]['tokens']


    result = []
    for item in dParse:
        rel = item['dep']
        left = item['dependentGloss']
        right = item['governorGloss']
        leftIndex = item['dependent']
        rightIndex = item['governor']
        # rel, left, right, leftIndex, rightIndex = item
        # print type(left), type(leftIndex), type(tokens[leftIndex]['characterOffsetBegin'])
        leftIndex -= 1
        rightIndex -= 1
        left += '{' + str(tokens[leftIndex]['characterOffsetBegin']) + ' ' + str(
            tokens[leftIndex]['characterOffsetEnd']) + ' ' + str(leftIndex + 1) + '}'
        right += '{' + str(tokens[rightIndex]['characterOffsetBegin']) + ' ' + str(
            tokens[rightIndex]['characterOffsetEnd']) + ' ' + str(rightIndex + 1) + '}'
        result.append([rel, left, right])
    print('depParse', result)
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
    # print tokensWithIndices

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
        # print True
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


nlp = StanfordNLP()

if __name__ == '__main__':
    parsetext = nlp.parse('I love China.')

    print(json.dumps(parsetext, indent=2))
    print(ner(parsetext))
    print(posTag(parsetext))
    print(lemmatize(parsetext))
    dep = dependencyParseAndPutOffsets(parsetext)
    print(findParents(dep, 1 + 1, 'love'))
    print(findChildren(dep, 1 + 1, 'love'))
    print(len(parsetext['sentences']))
