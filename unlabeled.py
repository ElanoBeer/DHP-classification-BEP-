# ---------------------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Load TQDM to Show Progress Bars #
from tqdm import tqdm

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

# warning avoidance
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

# hand-crafted functions
from pre_processing import *
from utilities import average_embeddings, random_shuffle
from model_settings import *
from active_learning import uncertainty_batch_teaching

# --------------------------------------------------------------------------------------------------------
# Implement uncertainty sampling on to predict on the unlabeled
# --------------------------------------------------------------------------------------------------------

# Find the best learner for the initial prediction
acc_lst, f1_lst, learners = uncertainty_batch_teaching(
    w2v_models=w2v_models,
    models=[lgcv, rfc, lsvc],
    X=labeled_lemma_df["text_clean"],
    y=labeled_lemma_df["label"],
    max_samples=len(labeled_lemma_df),
    smoothed=True,
)


def batch_teaching(
    w2v_model, model, X, y, unlabeled_X, max_samples: int, init_size=20, batch_size=20
):
    """_summary_

    Args:
        w2v_model (_type_): _description_
        model (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_
        unlabeled_X (_type_): _description_
        max_samples (int): _description_
        init_size (int, optional): _description_. Defaults to 20.
        batch_size (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """

    labeled_acc_lst = []
    labeled_f1_lst = []
    acc_lst = []
    f1_lst = []
    query_set = []

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=5
    )

    # Average word embeddings
    X_train_vect_avg, X_test_vect_avg = average_embeddings(w2v_model, X_train, X_test)

    # Initial labeled samples
    # Randomly oversample as SMOTE would not have enough neighbours
    X_initial, y_initial = oversampler.fit_resample(
        X_train_vect_avg[0:20], y_train[0:20]
    )
    X_initial, y_initial = random_shuffle(X_initial, y_initial)

    # Class balancing with SMOTE
    X_over, y_over = smote.fit_resample(X_train_vect_avg, y_train)
    X_train, y_train, X_test, y_test = map(
        np.array, (X_over, y_over, X_test_vect_avg, y_test)
    )

    learner = ActiveLearner(
        estimator=model,
        query_strategy=uncertainty_sampling,
        X_training=list(X_initial),
        y_training=list(y_initial),
    )

    # Labeled training
    model_acc_lst = []
    model_f1_lst = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        current_batch_size = init_size
        while current_batch_size <= max_samples:
            if model != lsvc:
                query_idx, _ = learner.query(X_train, n_instances=current_batch_size)
            else:
                # Calculate the distances from the decision boundary
                decision_values = model.decision_function(X_train)
                distances = np.abs(decision_values) / np.linalg.norm(model.coef_)
                query_idx = np.argsort(distances)[:current_batch_size]

            # teach only the new entries
            learner.teach(X_train[query_idx], y_train[query_idx], only_new=True)
            y_pred = learner.predict(X_test_vect_avg)
            model_acc_lst.append(accuracy_score(y_test, y_pred))
            model_f1_lst.append(f1_score(y_test, y_pred))

            current_batch_size += batch_size

        # Append model performances seperately
        labeled_acc_lst.append(model_acc_lst)
        labeled_f1_lst.append(model_f1_lst)

    # Unlabeled training
    unlabeled_vect_avg, _ = average_embeddings(w2v_model, unlabeled_X, X_test)
    unlabeled_pred = learner.predict(unlabeled_vect_avg)

    X_train, y_train = map(np.array, (unlabeled_vect_avg, unlabeled_pred))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        current_batch_size = init_size
        while current_batch_size <= len(X_train):
            if model != lsvc:
                query_idx, _ = learner.query(X_train, n_instances=current_batch_size)
            else:
                decision_values = model.decision_function(X_train)
                distances = np.abs(decision_values) / np.linalg.norm(model.coef_)
                query_idx = np.argsort(distances)[:current_batch_size]

            # teach only the new entries
            learner.teach(X_train[query_idx], y_train[query_idx], only_new=True)
            query_set.append(query_idx)
            y_pred = learner.predict(X_test_vect_avg)
            acc_lst.append(accuracy_score(y_test, y_pred))
            f1_lst.append(f1_score(y_test, y_pred))

            current_batch_size += batch_size

    final_pred = learner.predict(unlabeled_vect_avg)

    return acc_lst, f1_lst, final_pred, query_set


acc_lst, f1_lst, us_y_pred, query_set = batch_teaching(
    w2v_model_1,
    lgcv,
    labeled_lemma_df["text_clean"],
    labeled_lemma_df["label"],
    unlabeled_lemma_df["text_clean"],
    max_samples=500,
)

# --------------------------------------------------------------------------------------------------------
# Perform standard prediction procedure
# --------------------------------------------------------------------------------------------------------

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    labeled_lemma_df["text_clean"],
    labeled_lemma_df["label"],
    test_size=0.3,
    random_state=5,
)

# Average word embeddings
X_train_vect_avg, X_test_vect_avg = average_embeddings(w2v_model_1, X_train, X_test)


# Class balancing with SMOTE
X_over, y_over = smote.fit_resample(X_train_vect_avg, y_train)
X_train, y_train, X_test, y_test = map(
    np.array, (X_over, y_over, X_test_vect_avg, y_test)
)

# Unlabeled training
unlabeled_vect_avg, _ = average_embeddings(
    w2v_model_1, unlabeled_lemma_df["text_clean"], X_test
)

lgcv.fit(X_train, y_train)
y_pred = lgcv.predict(unlabeled_vect_avg)


# --------------------------------------------------------------------------------------------------------
# Compare both methods
# --------------------------------------------------------------------------------------------------------

# Evaluate the unlabeled class ratio to the labeled class ratio
support = labeled_lemma_df["label"].value_counts()
uc_support = np.unique(us_y_pred, return_counts=True)
st_support = np.unique(y_pred, return_counts=True)

ratio = support[1] / sum(support)
uc_ratio = uc_support[1][1] / sum(uc_support[1])
st_ratio = st_support[1][1] / sum(st_support[1])

# Obtain the most uncertain entries
lemma_df.loc[query_set[0]]["text"]

# Evaluate the positive labels manually
us_predicted_df = pd.DataFrame({"text": unlabeled_lemma_df["text"], "label": us_y_pred})
pos_us_predicted_df = us_predicted_df[us_predicted_df["label"] == 1]

st_predicted_df = pd.DataFrame({"text": unlabeled_lemma_df["text"], "label": y_pred})
pos_st_predicted_df = st_predicted_df[st_predicted_df["label"] == 1]

# Evaluate different cases
diff_predicted_df = pd.concat(
    [pos_us_predicted_df, pos_st_predicted_df]
).drop_duplicates(keep=False, inplace=False)
