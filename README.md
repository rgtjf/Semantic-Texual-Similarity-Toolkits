# Baseline System of Low Resource Task

using stst to build a baseline system of low-resource task 


## Install

```
pip install stst

```


es test 2934/3071 95.54

ar test 1312/1449 90.55



## Step

- how to prepare data

```
how to write this file
1. define the example, like what you want to get from one example
2. init the data from different place, and return the train/dev/test set.abs
3. `write_to_json` and `read_from_json`. `read_from_json` return the example list of required data

- in the `main.py`
    only need to read_from_json
- in the 'definefeature.py'
    consider the form of the examples
```

- how to translate data to nn

