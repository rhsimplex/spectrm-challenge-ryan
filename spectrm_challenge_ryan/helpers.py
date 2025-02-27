import pandas as pd
from os import listdir
from os.path import join


class SpectrmLoader(object):
    """
    Convenience class for loading the Spectrm-formatted datasets
    """
    def __init__(self, path='challenge_data/', ext='txt'):
        """
        Get the filenames containing the data
        """
        self.data_files = listdir(path)
        self.ext = ext
        self.path = path

    def load(self):
        """
        Returns a dict of datasets, with filenames as keys, minus the extension
        """
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
