import nltk
from nltk.tokenize import RegexpTokenizer
import pandas as pd


class WordFrequency(object):

    def __init__(self):
        # define the stopwords
        self.stop_words = pd.Series(nltk.corpus.stopwords.words('english'))

    def get_unique_non_stop_words(self, text):
        # mash all dataframe text together, if necessary
        if type(text) == pd.core.frame.DataFrame:
            text = text.text.str.cat(sep=' ')

        # get all the words
        tokenizer = RegexpTokenizer(r'\w+')
        self.tokens = pd.Series(tokenizer.tokenize(text.lower()))
        # count them
        self.counts = self.tokens.value_counts()

        # get unique words which are not stopwords
        t = pd.Series(self.tokens)
        t = pd.Series(t.value_counts().index)
        mask = t.isin(self.stop_words)
        self.unique_non_stop_words = t[-mask]
