from spectrm_challenge_ryan.helpers import SpectrmLoader
from spectrm_challenge_ryan.models import WordFrequency
import pandas as pd
import numpy as np
import sys


def softmax(x):
    """
    Softmax normalizes a vector into a probability distribution.
    After applying the softmax, the vectors sums to unity.
    """
    sf = np.exp(x)
    sf = sf/np.sum(sf, axis=0)
    return sf


def run(test_dialogs_file='challenge_data/test_dialogs.txt',
        test_missing_file='challenge_data/test_missing.txt'):
    loader = SpectrmLoader()
    """
    Load data, train on a corpus of words from the training data,
    then predict on `test_missing_file`, replacing the dummy labels
    with the predicted ones.
    """
    print 'Loading data.'
    test_dialogs = loader.spectrm_file_to_df(test_dialogs_file)
    print test_dialogs_file + ' loaded.'
    test_missing = loader.spectrm_file_to_df(test_missing_file)
    print test_missing_file + ' loaded.'

    print 'Generating corpus.'
    wf_corpus = WordFrequency()
    wf_corpus.get_unique_non_stop_words(test_dialogs)

    print 'Generating weights.'
    wf_corpus.generate_weights()
    labels = test_dialogs.index.unique()

    print 'Generating model, processing input...'
    input_mat = {}
    target_mat = {}
    wf_input = WordFrequency()
    wf_target = WordFrequency()
    cnt = 0
    # generate model
    for i, label in enumerate(labels):
        # get all the unique stopwords for training and missing
        wf_input.get_unique_non_stop_words(test_dialogs.ix[label])
        wf_target.get_unique_non_stop_words(test_missing.iloc[i].text)

        # generate appropriate vectors, including normalization
        input_mat[label] = wf_corpus.unique_non_stop_words.\
            isin(wf_input.unique_non_stop_words)
        input_mat[label].index = wf_corpus.unique_non_stop_words.values
        norm = np.linalg.norm(input_mat[label])
        if norm > 0.0:
            input_mat[label] = input_mat[label]/norm
        target_mat[label] = wf_corpus.unique_non_stop_words.\
            isin(wf_target.unique_non_stop_words)
        target_mat[label].index = wf_corpus.unique_non_stop_words.values
        norm = np.linalg.norm(target_mat[label])
        if norm > 0.0:
            target_mat[label] = target_mat[label]/norm

        cnt += 1
        if cnt % 1000 == 0:
            print 'Records processed: ' + str(cnt)

    print 'Converting to dataframes.'
    train_dialog_vectors = pd.DataFrame(input_mat).T.astype('float')
    train_missing_vectors = pd.DataFrame(target_mat).T.astype('float')
    # delete intermediate data structures
    del(input_mat)
    del(target_mat)

    print 'Doing classification.'
    t_weighted = np.dot(np.dot(train_dialog_vectors.values, wf_corpus.W),
                        train_missing_vectors.values.T)
    t_weighted = np.apply_along_axis(softmax, 0, t_weighted)

    selections = np.argmax(t_weighted, axis=0)
    new_index = labels[selections]

    print 'Writing output.'
    test_missing_done = pd.DataFrame(test_missing.values, index=new_index,
                                     columns=['missing'])
    test_missing_done.reindex(new_index)
    # because of the unusual seperator, we can't use the builtin .to_csv
    with open('test_missing_with_predictions.txt', 'w') as f:
        for i, label in enumerate(test_missing_done.index):
            f.write(label + ' +++$+++ '
                    + test_missing_done.iloc[i]['missing'] + '\n')

    print 'Done!'

if __name__ == '__main__':
    """
    This script should be called as
    python match_dialogs.py path/to/test_dialogs.txt path/to/test_missing.txt
    and write the predicted conversation numbers for all missing lines to a
    file named test_missing_with_predictions.txt
    """
    # if called with file names, load data from there else load from default
    # location / output an error
    if len(sys.argv) == 1:
        print 'Attempting to run from default locations'
        run()
    elif len(sys.argv) > 2:
        test_dialogs_file, test_missing_file = sys.argv[1], sys.argv[2]
        run(test_dialogs_file, test_missing_file)
    else:
        print ("please call this script with `python match_dialogs.py"
               "path/to/test_dialogs.txt path/to/test_missing.txt`")
