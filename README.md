# Spectrm Challenge

My entry to the [Spectrm Challenge](https://github.com/cod3licious/spectrm-challenge).  Uses a very simple word-frequencey model and achieves recall rates around 5-8%. For an explanation on how it works, [see the Jupyter notebook.](https://github.com/rhsimplex/spectrm-challenge-ryan/blob/master/Word%20Frequency%20Model.ipynb)

## Setup

`spectrm-challenge-ryan` only requires basic scientific python dependencies (numpy, scipy, pandas, matplotlib, nltk).  I recommend using a pre-packaged distribution like Anaconda for multiprocessing/memory efficiency. However, you can use `pip` to set up the dependencies:

```bash
$ pip install -r requirements.txt
```

## Running the model

Note: Though the matrices are pretty sparse, I used the regular `numpy` matrix implementation.  That means generating a model can be quite memory intensive.  All my tests were on a desktop with 10 cores and 16GB of memory.  A nice improvement would to have this code use the `scipy` sparse matrix implementation.

To run the model on the unlabeled examples, simply run:

```bash
$ python match_dialogs.py
```

It knows the default locations.  If you want to run on other datasets, for instance the training set, you can specify:

```bash
$ python match_dialogs.py challenge_data/train_dialogs.txt challenge_data/train_missing.txt
```
