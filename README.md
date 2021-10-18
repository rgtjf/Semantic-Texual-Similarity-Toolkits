# Semantic Textual Similarity Toolkits

[![Gitter](https://badges.gitter.im/owner/repo.png)](https://gitter.im/ecnunlp/Lobby?)

This is the code by [ECNU team](http://aclweb.org/anthology/S/S17/S17-2028.pdf) submitted to SemEval STS Task.

[slides]()

## Installation

```bash
# download the repo
git clone https://github.com/rgtjf/Semantic-Texual-Similarity-Toolkits.git
# download the dataset and stanford CoreNLP tools
sh download.sh
# run the demo
python demo.py
```

## Results

you can configure `sts_model.py` to see the performance of different features on STSBenchmark dataset.

### STSBenchmark

| Methods                | Dev      | Test     |
|------------------------|----------|----------|
| RF                     | 0.8333   | 0.7993   |
| GB                     | 0.8356   | 0.8022   |
| EN-seven               | 0.8466   | 0.8100   |
| ---------------------- | -------- | -------- |
| aligner                | 0.6991   | 0.6379   |
| idf_aligner            | 0.7969   | 0.7622   |
| BOWFeature-True        | 0.7584   | 0.6472   |
| BOWFeature-False       | 0.7788   | 0.6874   |
| nGramOverlapFeature    | 0.7817   | 0.7453   |
| BOWFeature             | 0.7639   | 0.6847   |
| AlignmentFeature       | 0.8163   | 0.7748   |
| WordEmbeddingFeature   | 0.8011   | 0.7128   |


### Reference

[STSBenchmark board](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)


## Contacts
Any questions, please feel free to contact us:
rgtjf1 AT 163 DOT com

## Citation
If you find this responsity helpful, please cite our paper.
```
@inproceedings{tian-etal-2017-ecnu,
    title = "{ECNU} at {S}em{E}val-2017 Task 1: Leverage Kernel-based Traditional {NLP} features and Neural Networks to Build a Universal Model for Multilingual and Cross-lingual Semantic Textual Similarity",
    author = "Tian, Junfeng  and
      Zhou, Zhiheng  and
      Lan, Man  and
      Wu, Yuanbin",
    booktitle = "Proceedings of the 11th International Workshop on Semantic Evaluation ({S}em{E}val-2017)",
    year = "2017",
    url = "https://aclanthology.org/S17-2028",
    pages = "191--197"
}
```

