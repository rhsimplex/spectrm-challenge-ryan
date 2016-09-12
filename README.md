# Spectrm Challenge

My entry to the Spectrm Challenge.  Uses a very simple word-frequencey model and achieves recall rates around 5-8%.

## Setup

`spectrm-challenge-ryan` only requires basic scientific python dependencies (numpy, scipy, pandas, matplotlib, nltk).  I recommend using a pre-packaged distribution like Anaconda for multiprocessing/memory efficiency. However, you can use `pip` to set up the dependencies:

```bash
$ pip install -r requirements.txt
```

## Running the model

Note: Though the matrices are in quite sparse, I used the regular `numpy` matrix implementation.  That means generating a model can be quite memory intensive.  All my tests were on a desktop with 10 cores and 16GB of memory.  A nice improvement would to have this code use the `scipy` spare matrix implementation.
