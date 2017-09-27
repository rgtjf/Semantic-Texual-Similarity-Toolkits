# Semantic Textual Similarity Toolkits

A light version of STST

Our goal:
1. easy and fast to build a nlp system
2. it's easy to use a library to be integrated into other systems.
3. and so on.



## How to use it?
```python
import stst

classifier = stst.Classifier('classifier')

model = stst.Model('model', classifier)

class DefineFeature(stst.Feature):
    def extract(self):
        pass

class DefineData(stst.Data):

    def __init__(self):
        pass

    def load_from_file():
        pass

    def __iter__(self):
        pass

class DefineEval(stst.Eval):

    def measure(self, a, b):
        pass


model.add(Define(Feature))
train_data = {文本/数字} field


model.train_model(train_data)
model.test_model(test_data)


```


## Installation
1. download the repo
2. python
    - requirements.txt
    - Usage
    ```
    $ pip install -r requirements.txt
3. Run

    ```
    cd test
    python train.py
    ```