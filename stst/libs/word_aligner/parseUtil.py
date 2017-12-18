# coding: utf8


def ner_word_annotator(parse_sent):
    """
    Return NER words
    Args:
        parse_sent: json

    Returns:
        List of NER words: [[char_begin, char_end], word_index, word, ner]
    """
    res = []
    word_index = 1
    for i in range(len(parse_sent['sentences'][0]['tokens'])):
        tag = [[
            parse_sent['sentences'][0]['tokens'][i]['characterOffsetBegin'],
            parse_sent['sentences'][0]['tokens'][i]['characterOffsetEnd']
            ],
            word_index,
            parse_sent['sentences'][0]['tokens'][i]['word'],
            parse_sent['sentences'][0]['tokens'][i]['ner']]
        word_index += 1
        if tag[3] != 'O':
            res.append(tag)
    return res


def ner(parse_sent):
    """
    ner
    Args:
        parse_sent:
    Returns:
    """
    nerWordAnnotations = ner_word_annotator(parse_sent)

    namedEntities = []

    # current lists
    currentNE = []
    currentCharacterOffsets = []
    currentWordOffsets = []

    for i in range(len(nerWordAnnotations)):

        if i == 0 or nerWordAnnotations[i][3] == nerWordAnnotations[i - 1][3] \
                and nerWordAnnotations[i][1] == nerWordAnnotations[i - 1][1] + 1:
            currentNE.append(nerWordAnnotations[i][2])
            currentCharacterOffsets.append(nerWordAnnotations[i][0])
            currentWordOffsets.append(nerWordAnnotations[i][1])
        else:
            # add the previous set into namedEntities
            namedEntities.append([currentCharacterOffsets, currentWordOffsets, currentNE, nerWordAnnotations[i - 1][3]])
            currentNE = [nerWordAnnotations[i][2]]
            currentCharacterOffsets = [nerWordAnnotations[i][0]]
            currentWordOffsets = [nerWordAnnotations[i][1]]

        # add the last ner word
        if i == len(nerWordAnnotations) - 1:
            namedEntities.append([currentCharacterOffsets, currentWordOffsets, currentNE, nerWordAnnotations[i][3]])

    return namedEntities


def posTag(parseResult):
    """
    Args:
        parseResult:
    Return:

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
                parseResult['sentences'][0]['tokens'][i]['pos']
        ]
        wordIndex += 1
        res.append(tag)

    return res


def lemmatize(parseResult):
    """

    Args:
        parseResult:

    Returns:

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


def dependencyParseAndPutOffsets(parseResult):
    """
    Args:
        parseResult:

    Returns:
        (rel, left{charStartOffset, charEndOffset, wordNumber}, right{charStartOffset, charEndOffset, wordNumber})
        dependency parse of the sentence where each item is of the form
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
        left += '{' + str(tokens[leftIndex]['characterOffsetBegin']) + ' ' + str(
            tokens[leftIndex]['characterOffsetEnd']) + ' ' + str(leftIndex + 1) + '}'
        right += '{' + str(tokens[rightIndex]['characterOffsetBegin']) + ' ' + str(
            tokens[rightIndex]['characterOffsetEnd']) + ' ' + str(rightIndex + 1) + '}'
        result.append([rel, left, right])

    return result


def findParents(dependencyParse, wordIndex, word):
    """
    Note: word index assumed to be starting at 1
    the third parameter is needed because of the collapsed representation of the dependencies...
    Args:
        dependencyParse:
        wordIndex:
        word:

    Returns:

    """
    tokensWithIndices = ((int(item[2].split('{')[1].split('}')[0].split(' ')[2]), item[2].split('{')[0]) for item in dependencyParse)
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


def findChildren(dependencyParse, wordIndex, word):
    """
    Note: word index assumed to be starting at 1
    the third parameter is needed because of the collapsed representation of the dependencies...
    Args:
        dependencyParse:
        wordIndex:
        word:

    Returns:

    """

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


if __name__ == '__main__':
    parsetext = """{"sentences": [{"tokens": [{"word": "I", "lemma": "I", "pos": "PRP", "index": 1, "originalText": "I", "before": "", "after": " ", "characterOffsetBegin": 0, "ner": "O", "characterOffsetEnd": 1}, {"word": "love", "lemma": "love", "pos": "VBP", "index": 2, "originalText": "love", "before": " ", "after": " ", "characterOffsetBegin": 2, "ner": "O", "characterOffsetEnd": 6}, {"word": "China", "lemma": "China", "pos": "NNP", "index": 3, "originalText": "China", "before": " ", "after": "", "characterOffsetBegin": 7, "ner": "LOCATION", "characterOffsetEnd": 12}, {"word": ".", "lemma": ".", "pos": ".", "index": 4, "originalText": ".", "before": "", "after": "", "characterOffsetBegin": 12, "ner": "O", "characterOffsetEnd": 13}], "collapsed-ccprocessed-dependencies": [{"dependentGloss": "love", "dependent": 2, "governorGloss": "ROOT", "governor": 0, "dep": "ROOT"}, {"dependentGloss": "I", "dependent": 1, "governorGloss": "love", "governor": 2, "dep": "nsubj"}, {"dependentGloss": "China", "dependent": 3, "governorGloss": "love", "governor": 2, "dep": "dobj"}, {"dependentGloss": ".", "dependent": 4, "governorGloss": "love", "governor": 2, "dep": "punct"}], "collapsed-dependencies": [{"dependentGloss": "love", "dependent": 2, "governorGloss": "ROOT", "governor": 0, "dep": "ROOT"}, {"dependentGloss": "I", "dependent": 1, "governorGloss": "love", "governor": 2, "dep": "nsubj"}, {"dependentGloss": "China", "dependent": 3, "governorGloss": "love", "governor": 2, "dep": "dobj"}, {"dependentGloss": ".", "dependent": 4, "governorGloss": "love", "governor": 2, "dep": "punct"}], "index": 0, "basic-dependencies": [{"dependentGloss": "love", "dependent": 2, "governorGloss": "ROOT", "governor": 0, "dep": "ROOT"}, {"dependentGloss": "I", "dependent": 1, "governorGloss": "love", "governor": 2, "dep": "nsubj"}, {"dependentGloss": "China", "dependent": 3, "governorGloss": "love", "governor": 2, "dep": "dobj"}, {"dependentGloss": ".", "dependent": 4, "governorGloss": "love", "governor": 2, "dep": "punct"}]}]}"""
    import json
    # l love China.
    parsetext = json.loads(parsetext)

    print(ner(parsetext))
    print(posTag(parsetext))
    print(lemmatize(parsetext))
    dep = dependencyParseAndPutOffsets(parsetext)
    print(findParents(dep, 1 + 1, 'love'))
    print(findChildren(dep, 1 + 1, 'love'))
    print(len(parsetext['sentences'][0]['tokens']))
