# A Word Aligner for English

This is a word aligner for English: given two English sentences, it aligns related words in the two sentences. It exploits the semantic and contextual similarities of the words to make alignment decisions.


## Ack
Initially, this is a fork of <i>[ma-sultan/monolingual-word-aligner](https://github.com/ma-sultan/monolingual-word-aligner)</i>, the aligner presented in [Sultan et al., 2015](http://aclweb.org/anthology/S/S15/S15-2027.pdf) that has been very successful in [SemEval STS (Semantic Textual Similarity) Task](http://alt.qcri.org/semeval2017/task1/) in recent years.


## SubTree Module

```bash
git remote add wordaligner https://github.com/rgtjf/monolingual-word-aligner.git

git subtree add --prefix=PATH --squash wordaligner dev-lib
```

## Usage

```python
import word_aligner

features, infos = word_aligner.align_feats(parse_sa, parse_sb)
```
