# --------------------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from sklearn.metrics import recall_score

# --------------------------------------------------------------------------------------------------------
# Create functions to use throughout simulations and analysis
# ---------------------------------------------------------------------------------------------------------


def average_embeddings(w2v_model, X_train, X_test):
    """Uses a trained word embeddings model on the entire corpus
       to create uniform w2v embeddings based on averaging.

    Args:
        w2v_model (class): the trained gensim word2vec model
        X_train (Series): the train series after performing the train_test_split
        X_test (Series): the test series after performing the train_test_split

    Returns:
        X_train_vect_avg (list): lists of average word2vec embeddings for platform descriptions in the training set
        X_test_vect_avg (list): lists of average word2vec embeddings for platform descriptions in the testing set
    """

    words = set(w2v_model.wv.index_to_key)
    X_train_vect = np.array(
        [np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_train],
        dtype=object,
    )
    X_test_vect = np.array(
        [np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_test],
        dtype=object,
    )

    # Compute sentence vectors by averaging the word vectors for the words contained in the sentence
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(100, dtype=float))

    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))
    return X_train_vect_avg, X_test_vect_avg


def fnr(y, predictions):
    """A function that calculates the False Negative Rate."""

    fnr = 1 - recall_score(y, predictions)
    return fnr


def random_shuffle(X, y):
    """A function that randomly shuffles two lists in similar fashion."""
    initial_lst = list(zip(X, list(y)))
    shuffled_lst = shuffle(initial_lst, random_state=5)
    X, y = zip(*shuffled_lst)
    return X, y
