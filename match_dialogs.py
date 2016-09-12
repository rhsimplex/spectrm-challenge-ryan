from spectrm_challenge_ryan.helpers import SpectrmLoader
from spectrm_challenge_ryan.models import WordFrequency
import pandas as pd
import numpy as np
import sys


def run(test_dialogs_file='challenge_data/test_dialogs.txt',
        test_missing_file='challenge_data/test_missing.txt',
        limit=None):
    """
    Load data, train on a corpus of words from the training data,
    then predict on `test_missing_file`, replacing the dummy labels
    with the predicted ones.

    Args:
        test_dialogs_file (str): path to dialogs
        test_missing_file (str): path to file with missing lines
        limit (int): Limit the number of dialogs. Can be useful
            if memory is constrained. Default None (use all)
    """
    loader = SpectrmLoader()
    print 'Loading data...'
    test_dialogs = loader.spectrm_file_to_df(test_dialogs_file)
    print test_dialogs_file + ' loaded.'
    test_missing = loader.spectrm_file_to_df(test_missing_file)
    print test_missing_file + ' loaded.'
    if limit:
        labels = test_dialogs.index.unique()[:limit]
        test_dialogs = test_dialogs.ix[labels]
    else:
        labels = test_dialogs.index.unique()
    print ('Using ' + str(labels.shape[0]) + ' training dialogs and'
           ' attempting to classify ' + str(test_missing.shape[0]) +
           ' examples.')
    print 'Generating corpus...'
    wf_corpus = WordFrequency()
    wf_corpus.get_unique_non_stop_words(test_dialogs)
    print ('Using ' + str(wf_corpus.unique_non_stop_words.shape[0])
           + ' unique words.')

    print 'Generating weights...'
    wf_corpus.generate_weights()

    print 'Generating model, processing input...'
    input_mat = {}
    wf_input = WordFrequency()
    cnt = 0

    # generate model
    for label in labels:
        # get all the unique stopwords for training
        wf_input.get_unique_non_stop_words(test_dialogs.ix[label])

        # generate a vector with 1s where a word matches a word in the
        # corpus, zero elsewhere
        input_mat[label] = wf_corpus.unique_non_stop_words.\
            isin(wf_input.unique_non_stop_words)

        # index them to the words themselves -- useful for debugging
        input_mat[label].index = wf_corpus.unique_non_stop_words.values
        # divide by the norm, so dialogs with many words don't have
        # an advantage
        norm = np.linalg.norm(input_mat[label])
        if norm > 0.0:
            input_mat[label] = input_mat[label]/norm

        cnt += 1
        if cnt % 1000 == 0:
            print 'Records processed: ' + str(cnt)

    print 'Converting to dataframes...'
    train_dialog_vectors = pd.DataFrame(input_mat).T.astype('float')
    # delete intermediate data structures
    del(input_mat)

    print 'Doing classification...'
    wf_target = WordFrequency()
    cnt = 0
    with open('test_missing_with_predictions.txt', 'w') as f:
        # load_mat is the product of the normalized dialog vectors times the
        # word weights, which are inversely proportional to the word frequency
        # in the corpus
        load_mat = np.dot(train_dialog_vectors.values, wf_corpus.W)
        for i in range(test_missing.shape[0]):
            # we do this in a loop rather than by direct matrix multiplication;
            # it's slower, but saves us some memory.
            #
            # Get the unique stop words out of the target (missing line)
            wf_target.get_unique_non_stop_words(test_missing.iloc[i].text)
            # Compose the vector out of words present in the line, and
            # normalize
            target_vec = wf_corpus.unique_non_stop_words\
                .isin(wf_target.unique_non_stop_words)
            norm = np.linalg.norm(target_vec)
            if norm > 0.0:
                target_vec = target_vec/norm
            # score every weighted input (dialog) against this missing line
            scores = np.dot(load_mat, target_vec)
            # the maxiumum score is our best guess to which corresponding row
            # in train_dialog_vectors that our missing line belongs to
            pos_max = np.argmax(scores)
            # get the corresponding label
            winner = train_dialog_vectors.iloc[pos_max].name
            # finally, write our output
            f.write(winner + ' +++$+++ ' + test_missing.iloc[i].text + '\n')
            cnt += 1
            if cnt % 500 == 0:
                print 'Classifications done: ' + str(cnt)

    print 'Done! Output in test_missing_with_predictions.txt'

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
