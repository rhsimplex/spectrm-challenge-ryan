import pandas as pd
from os import listdir
from os.path import join


class SpectrmLoader(object):

    def __init__(self, path='challenge_data/', ext='txt'):
        self.data_files = listdir(path)
        self.ext = ext
        self.path = path

    def load(self):
        return {label[:-(len(self.ext) + 1)]:
                self.spectrm_file_to_df(join(self.path, label))
                for label in self.data_files}

    @staticmethod
    def spectrm_file_to_df(path, sep=' \+\+\+\$\+\+\+ ', names=['id', 'text']):
        """
        Load a Spectrm-type input file into a pandas dataframe
        """
        return pd.read_csv(path, sep=sep, engine='python', header=None,
                           names=names, index_col=names[0])
