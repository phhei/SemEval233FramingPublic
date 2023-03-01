# Team ACCEPT at SemEval-2023 Task 3: An Ensemble-based Approach to Multilingual Framing Detection

with [SemEval 2023 Task 3
"Detecting the ~~Genre,~~ the Framing, ~~and the Persuasion Techniques~~ in Online News in a Multi-lingual Setup"](https://propaganda.math.unipd.it/semeval2023task3/index.html)

Here we present our comprehensive ensemble-framework which you can call with ``mainFrame.py [--help]``. Please use Python 3.9 and run ``python -m pip install -r requirements.txt`` beforehand.

## Cite us when you want to use our framework

````text
@inproceedings{heinisch-etal-2023-ensemble,
    title={Team ACCEPT at SemEval-2023 Task 3: An Ensemble-based Approach to Multilingual Framing Detection},
    author={Heinisch, Philipp and Plenz, Moritz and Frank, Anette and Cimiano, Philipp},
    booktitle = {Proceedings of the 17th International Workshop on Semantic Evaluation},
    series = {SemEval 2023},
    year = {2023},
    address = {Toronto, Canada},
    month = {July},
}
````

## Implement further methods

1. Instantiate
   1. a data-preprocessor: ``pipeline/preprocessing/FrameDatasetInterface.py``
   1. an encoder: ``pipeline/encoder/EncoderInterface.py``
   1. (if you need a fancy additional aggregator when text chunking is applied: ``pipeline/aggregator/AggregatorInterface.py``)
1. add your new method in ``const.py``
1. if you need additional libraries, update ``requirements.txt`` please!
1. enjoy - ensemble- combine :)
